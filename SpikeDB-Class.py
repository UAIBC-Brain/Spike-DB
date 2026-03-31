import os
import sys
import types
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from x_transformers import Encoder
from SNN_fMRI import SNN_Model

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

def compute_sen_spe(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    SEN = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    SPE = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    return round(ACC, 5), round(SEN, 5), round(SPE, 5), round(F1, 5)


class IJEPA_fMRI_classifier(nn.Module):
    def __init__(self, num_regions=90, seq_len=240, embed_dim=64, enc_depth=6, num_heads=8, num_classes=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Linear(seq_len, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_regions, embed_dim))
        self.post_emb_norm = nn.LayerNorm(embed_dim)
        self.token_lif = snn.Leaky(beta=BETA)

        self.encoder = Encoder(dim=embed_dim, heads=num_heads, depth=enc_depth, layer_dropout=0.0)
        self.encoder_lif = snn.Leaky(beta=BETA)

        self.head_norm = nn.LayerNorm(embed_dim)
        self.pool_lif = snn.Leaky(beta=BETA)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        self.hidden_lif = snn.Leaky(beta=BETA)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self.classifier_lif = snn.Leaky(beta=BETA)

    def forward(self, x):
        """
        x: [B, num_regions, seq_len]
        """
        x = self.token_embed(x) + self.pos_embedding     # [B, R, D]
        x = self.post_emb_norm(x)                        # [B, R, D]
        x, _ = self.token_lif(x)
        x = self.encoder(x)                              # [B, R, D]
        x, _ = self.encoder_lif(x)
        x = self.head_norm(x)                            # [B, R, D]  <-- LN 
        x = x.mean(dim=1)                                # [B, D]
        x, _ = self.pool_lif(x)
        x = self.hidden(x)
        x, _ = self.hidden_lif(x)
        x = self.dropout(x)
        logits = self.head(x)                            # [B, num_classes]
        logits, _ = self.classifier_lif(logits)
        return logits

class IJEPA_fMRI_Lit(pl.LightningModule):
    def __init__(self, dataset_path, label_path, num_regions=90, seq_len=240, embed_dim=64, enc_heads=8, enc_depth=6,
                 lr=1e-4, weight_decay=0.05, num_classes=2, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = IJEPA_fMRI_classifier(
            num_regions=num_regions, seq_len=seq_len, embed_dim=embed_dim,
            enc_depth=enc_depth, num_heads=enc_heads, num_classes=num_classes, dropout=dropout
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        ACC2, SEN, SPE, F1 = compute_sen_spe(preds, y)
        ACC1 = (preds == y).float().mean().item()
        # ACC = (logits.argmax(dim=1) == y).float().mean()
        ACC1 = round(ACC1, 5)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_ACC1", ACC1, prog_bar=True, on_epoch=True)
        self.log("train_ACC2", ACC2, prog_bar=True, on_epoch=True)
        self.log("train_SEN", SEN, prog_bar=True, on_epoch=True)
        self.log("train_SPE", SPE, prog_bar=True, on_epoch=True)
        self.log("train_F1", F1, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        ACC2, SEN, SPE, F1 = compute_sen_spe(preds, y)
        ACC1 = (preds == y).float().mean().item()
        # ACC = (logits.argmax(dim=1) == y).float().mean()
        ACC1 = round(ACC1, 5)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_ACC1", ACC1, prog_bar=True, on_epoch=True)
        self.log("val_ACC2", ACC2, prog_bar=True, on_epoch=True)
        self.log("val_SEN", SEN, prog_bar=True, on_epoch=True)
        self.log("val_SPE", SPE, prog_bar=True, on_epoch=True)
        self.log("val_F1", F1, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class FMRI_Classification_Dataset(Dataset):
    def __init__(self, data_path, label_path, zscore=False, spike_model=None, device="cpu"):
        mat_data = sio.loadmat(data_path)
        mat_label = sio.loadmat(label_path)

        X = mat_data["Data"].astype(np.float32)     # [N, R, T]
        y = mat_label["Y"]

        if y.ndim > 1:
            if y.shape[0] == 1 or y.shape[1] == 1:
                y = y.squeeze()
            else:
                y = y.argmax(axis=-1).squeeze()
        y = y.astype(np.int64)
        if y.min() == 1 and y.max() == 2:
            y = y - 1

        assert len(X) == len(y), f"The number of data and tags does not match: X={len(X)}, y={len(y)}"

        if zscore:
            mean = X.mean(axis=2, keepdims=True)
            std = X.std(axis=2, keepdims=True) + 1e-6
            X = (X - mean) / std

        self.X = X
        self.y = y
        self.spike = spike_model
        self.device = device

        if self.spike is not None:
            self.spike = self.spike.to(self.device)
            self.spike.eval()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()   # [R, T]
        x = x.unsqueeze(0).to(self.device)
        if self.spike is not None:
            with torch.no_grad():
                x = self.spike(x)
        x = x.squeeze(0).cpu()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class FMRIDataModule(pl.LightningDataModule):
    def __init__(self, data_path, label_path, batch_size=8, num_workers=2, shuffle=True, zscore=False, device="cpu"):
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.zscore = zscore
        self.device = device

    def setup(self, stage=None):
        snn_model = SNN_Model()
        full_dataset = FMRI_Classification_Dataset(
            self.data_path,
            self.label_path,
            zscore=self.zscore,
            spike_model=snn_model,
            device=self.device,
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    data_path = r"G:\XXX\XXX.mat"
    label_path = r"G:\XXX\XXX.mat"

    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    dm = FMRIDataModule(data_path, label_path, batch_size=2, num_workers=2, shuffle=True, zscore=False, device=runtime_device)

    model = IJEPA_fMRI_Lit(dataset_path=data_path,
                           label_path=label_path,
                           num_regions=90,
                           seq_len=240,
                           embed_dim=64,
                           enc_heads=8,
                           enc_depth=6,
                           lr=1e-4,
                           weight_decay=0.05,
                           num_classes=2,
                           dropout=0.1)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        max_epochs=30,
        callbacks=[LearningRateMonitor(logging_interval="step"), ModelSummary(max_depth=2)],
        gradient_clip_val=0.1,
    )
    trainer.fit(model, dm)
