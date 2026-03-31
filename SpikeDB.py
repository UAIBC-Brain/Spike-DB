import os
import math
import sys
import types
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from x_transformers import Encoder, Decoder
from SNN_fMRI import SNN_Model 
import copy

try:
    import importlib.metadata  # type: ignore
except ModuleNotFoundError:
    import importlib_metadata

    sys.modules["importlib.metadata"] = importlib_metadata


class _StatelessSurrogateLeaky(nn.Module):
    def __init__(self, beta=0.9, threshold=0.5, slope=10.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)

        mem = self.beta * mem + x
        surrogate = torch.sigmoid(self.slope * (mem - self.threshold))
        hard_spike = (mem >= self.threshold).float()
        spike = hard_spike.detach() - surrogate.detach() + surrogate
        reset_mem = mem * (1.0 - hard_spike.detach())
        return spike, reset_mem


try:
    import snntorch as snn
except ModuleNotFoundError:
    fallback_snntorch = types.ModuleType("snntorch")
    fallback_snntorch.Leaky = _StatelessSurrogateLeaky
    sys.modules.setdefault("snntorch", fallback_snntorch)
    import snntorch as snn


BETA = 0.9

# -------------------------
# Model components (adapted to fMRI tokens)
# -------------------------

class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()

        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)
        self.predict_lif = snn.Leaky(beta=BETA)

    def forward(self, context_encoding, target_masks):
       
        x = torch.cat((context_encoding, target_masks), dim=1)   # [B, Lc+Lt, D]
        x = self.predictor(x)   
        x, _ = self.predict_lif(x)
        l = x.shape[1]          
        return x[:, l - target_masks.shape[1]:, :]   

class IJEPA_fMRI_base(nn.Module):
    def __init__(self, num_regions=90, seq_len=240, embed_dim=64, enc_depth=8, pred_depth=6, num_heads=8, post_emb_norm=False, M=90, mode="train", layer_dropout=0.):
       
        super().__init__()
        self.num_regions = num_regions
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout

        self.token_embed = nn.Linear(seq_len, embed_dim)    

        self.pos_embedding = nn.Parameter(torch.randn(1, num_regions, embed_dim))   

        self.decoder_to_seq = nn.Linear(embed_dim, seq_len)  

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))     
        nn.init.trunc_normal_(self.mask_token, 0.02)          

        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()  
        self.norm = nn.LayerNorm(embed_dim)            
        self.token_lif = snn.Leaky(beta=BETA)
        self.teacher_lif = snn.Leaky(beta=BETA)
        self.student_lif = snn.Leaky(beta=BETA)
        self.mask_lif = snn.Leaky(beta=BETA)
        self.output_lif = snn.Leaky(beta=BETA)

        self.teacher_encoder = Encoder(dim=embed_dim, heads=num_heads, depth=enc_depth, layer_dropout=self.layer_dropout)   
        self.student_encoder = copy.deepcopy(self.teacher_encoder)      

        self.predictor = Predictor(embed_dim, num_heads, pred_depth)   

    @torch.no_grad()
    def get_teacher_full(self, x_emb):
       
        self.teacher_encoder = self.teacher_encoder.eval()    
        teacher_full = self.norm(self.teacher_encoder(x_emb))
        teacher_full, _ = self.teacher_lif(teacher_full)
        return teacher_full       

    @staticmethod
    def compute_firing_rate(spike_seq):
        return spike_seq.float().mean(dim=-1)

    def forward(self, x, M=None, exhaustive=True):
        
        if M is None:
            M = self.M
        B = x.shape[0]       # batch size
        R = self.num_regions

        x_emb = self.token_embed(x)          # [B, R, seq_len] -> [B, R, D]

        x_emb = x_emb + self.pos_embedding   # [B, R, D]

        x_emb = self.post_emb_norm(x_emb)
        x_emb, _ = self.token_lif(x_emb)

        if self.mode == 'test':
            # return student encoder full embeddings
            student_test = self.student_encoder(x_emb)
            student_test = self.norm(student_test)
            student_test, _ = self.student_lif(student_test)
            return student_test

        with torch.no_grad():
            teacher_full = self.get_teacher_full(x_emb)  # [B, R, D]

        student_full = self.student_encoder(x_emb)     # [B, R, D]
        student_full = self.norm(student_full)
        student_full, _ = self.student_lif(student_full)


        if exhaustive:
            target_indices = list(range(R))   
        else:
            target_indices = np.random.choice(R, size=min(M, R), replace=False).tolist()

        preds = []         
        teachers = []      

        for t in target_indices:
            teacher_target = teacher_full[:, t:t+1, :]   # [B,1,D]
            teachers.append(teacher_target)

            idx = [i for i in range(R) if i != t]
            context_encoding = student_full[:, idx, :]   # [B, R-1, D]

            target_mask = self.mask_token.repeat(B, 1, 1)        # [B,1,D]
            target_pos = self.pos_embedding[:, t:t+1, :]         # [1,1,D]
            target_mask = target_mask + target_pos              # [B,1,D]
            target_mask, _ = self.mask_lif(target_mask)

            pred = self.predictor(context_encoding, target_mask) # [B,1,D]
            preds.append(pred)

        pred_targets = torch.cat(preds, dim=1)    # [B, K, D]
        teacher_targets = torch.cat(teachers, dim=1)  # [B, K, D]
        pred_seq_current = self.decoder_to_seq(pred_targets)  # [B, K, seq_len]
        teacher_seq_current = self.decoder_to_seq(teacher_targets)  # [B, K, seq_len]

        pred_spike_seq, _ = self.output_lif(pred_seq_current)
        teacher_spike_seq, _ = self.output_lif(teacher_seq_current)

        target_seq = x[:, target_indices, :]
        pred_rate = self.compute_firing_rate(pred_spike_seq)
        teacher_rate = self.compute_firing_rate(teacher_spike_seq)
        target_rate = self.compute_firing_rate(target_seq)

        return {
            "pred_seq": pred_spike_seq,
            "teacher_seq": teacher_spike_seq,
            "target_seq": target_seq,
            "pred_rate": pred_rate,
            "teacher_rate": teacher_rate,
            "target_rate": target_rate,
            "target_indices": target_indices,
        }


def firing_rate_mse_loss(pred_rate, target_rate):
    return torch.mean((pred_rate - target_rate) ** 2)


class IJEPA_fMRI(pl.LightningModule):
    def __init__(
        self,
        dataset_path,       
        num_regions=90,      
        seq_len=240,        
        embed_dim=64,        
        enc_heads=8,         
        enc_depth=8,         
        decoder_depth=6,     
        lr=1e-4,             
        weight_decay=0.05,   
        M=None,              
        exhaustive=True,     
        m=0.996,             
        m_start_end=(0.996, 1.0),    
    ):
        super().__init__()
        self.save_hyperparameters()   
        if M is None:
            M = num_regions  

        self.model = IJEPA_fMRI_base(num_regions=num_regions, seq_len=seq_len, embed_dim=embed_dim,
                                     enc_depth=enc_depth, pred_depth=decoder_depth, num_heads=enc_heads, M=M)
      
        self.lr = lr
        self.weight_decay = weight_decay
        self.M = M
        self.exhaustive = exhaustive
        self.num_regions = num_regions

        self.criterion = firing_rate_mse_loss

        self.m = m
        self.m_start_end = m_start_end

        self.dataset_path = dataset_path

        self.val_sum_sq_error = None       
        self.val_n_elements = None         
        self.val_sum_targets = None        
        self.val_sum_targets_sq = None     

    def forward(self, x):
        return self.model(x, M=self.M, exhaustive=self.exhaustive)    

    def update_momentum(self, m):
        student_model = self.model.student_encoder
        teacher_model = self.model.teacher_encoder
        with torch.no_grad():
            for s_p, t_p in zip(student_model.parameters(), teacher_model.parameters()):
                t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)

    def training_step(self, batch, batch_idx):
        x = batch  # [B, R, T]
        outputs = self(x)
        loss = self.criterion(outputs["pred_rate"], outputs["target_rate"])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_rate_mse', loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_sum_sq_error = 0.0
        self.val_n_elements = 0
        self.val_sum_targets = 0.0
        self.val_sum_targets_sq = 0.0

    def validation_step(self, batch, batch_idx):
        x = batch
        outputs = self(x)
        pred_rate = outputs["pred_rate"]
        target_rate = outputs["target_rate"]
        loss = self.criterion(pred_rate, target_rate)
        # log batch loss as usual
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        pred_cpu = pred_rate.detach().cpu()
        target_cpu = target_rate.detach().cpu()

        diff = pred_cpu - target_cpu
        sq_err = (diff ** 2).sum().item()   
        n_elems = pred_cpu.numel()         

        self.val_sum_sq_error += sq_err
        self.val_n_elements += n_elems
        self.val_sum_targets += target_cpu.sum().item()
        self.val_sum_targets_sq += (target_cpu ** 2).sum().item()

        return loss

    def on_validation_epoch_end(self):
        if self.val_n_elements is None or self.val_n_elements == 0:
            return

        mse = self.val_sum_sq_error / float(self.val_n_elements)
        rmse = math.sqrt(mse)

        mean_targets = self.val_sum_targets / float(self.val_n_elements)
        
        ss_tot = self.val_sum_targets_sq - float(self.val_n_elements) * (mean_targets ** 2)
      
        r2 = 0.0 if ss_tot <= 0 else 1.0 - (self.val_sum_sq_error / ss_tot)

        # RSE = RMSE / std(y) ; std(y) = sqrt(ss_tot / n)
        if ss_tot <= 0:
            rse = float('inf')
        else:
            std_y = math.sqrt(ss_tot / float(self.val_n_elements))
            rse = rmse / (std_y + 1e-12)

        self.log('val_MSE', mse, prog_bar=True, on_epoch=True)
        self.log('val_RMSE', rmse, on_epoch=True)
        self.log('val_RSE', rse, prog_bar=True, on_epoch=True)
        self.log('val_R2', r2, prog_bar=True, on_epoch=True)

        self.print(f"VAL METRICS -- MSE: {mse:.6f}, RMSE: {rmse:.6f}, RSE: {rse:.6f}, R2: {r2:.6f}")

    def on_after_backward(self):
        self.update_momentum(self.m)
        if self.trainer is not None:
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            self.m += (self.m_start_end[1] - self.m_start_end[0]) / float(total_steps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)   
        if self.trainer is not None:
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer


class FMRI_Dataset(Dataset):
    def __init__(self, mat_path, stage='train', spike_model=None, device="cpu"):
        
        mat = sio.loadmat(mat_path)   
        arr = mat['Data']             
        self.Data = arr.astype(np.float32)   
        self.normalize = False       
        if self.normalize:
            mean = self.Data.mean(axis=2, keepdims=True)
            std = self.Data.std(axis=2, keepdims=True) + 1e-6
            self.Data = (self.Data - mean) / std

        self.spike = spike_model
        self.device = device

        if self.spike is not None:
            self.spike = self.spike.to(self.device)
            self.spike.eval()

    def __len__(self):
        return self.Data.shape[0]   

    def __getitem__(self, idx):

        x = self.Data[idx] 
        x = torch.from_numpy(x).float() 
        x = x.unsqueeze(0).to(self.device)  
        if self.spike is not None:
            with torch.no_grad():
                x = self.spike(x)  
        x = x.squeeze(0).cpu() 
        return x
        # return torch.from_numpy(self.Data[idx]).float()


class FMRIDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=8, num_workers=2, shuffle=True, device='cpu'):
        super().__init__()
        self.dataset_path = dataset_path     
        self.batch_size = batch_size         
        self.num_workers = num_workers       
        self.shuffle = shuffle               
        self.device = device

    def setup(self, stage=None):
        
        snn_model = SNN_Model()  
        
        full_dataset = FMRI_Dataset(self.dataset_path, spike_model=snn_model, device=self.device)     

        train_size = int(0.8 * len(full_dataset))      
        val_size = len(full_dataset) - train_size      
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,            
            batch_size=self.batch_size,    
            shuffle=self.shuffle,         
            num_workers=self.num_workers,  
            pin_memory=True                
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,              
            batch_size=self.batch_size,    
            shuffle=False,                 
            num_workers=self.num_workers, 
            pin_memory=True               
        )


if __name__ == "__main__":
  
    dataset_path = r'G:\XXX\XXX.mat'

    dm = FMRIDataModule(dataset_path=dataset_path, batch_size=8, num_workers=2, shuffle=True)  

    model = IJEPA_fMRI(dataset_path=dataset_path,
                       num_regions=90,
                       seq_len=240,
                       embed_dim=64,        
                       enc_heads=8,         
                       enc_depth=6,         
                       decoder_depth=4,     
                       lr=1e-4,             
                       weight_decay=0.05,   
                       M=90,                
                       exhaustive=True,     
                       m=0.996,             
                       m_start_end=(0.996, 1.0)    
                       )

    lr_monitor = LearningRateMonitor(logging_interval="step")    
    model_summary = ModelSummary(max_depth=2)                   

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',      
        devices=1,                                                      
        precision=16 if torch.cuda.is_available() else 32,              
        max_epochs=50,                                                  
        callbacks=[lr_monitor, model_summary],                          
        gradient_clip_val=.1,                                           
    )

    trainer.fit(model, dm)
