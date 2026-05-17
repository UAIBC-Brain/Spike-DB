import argparse
import copy
import math
import os
import sys
import types

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from x_transformers import Decoder, Encoder
except ModuleNotFoundError:
    class Encoder(nn.Module):
        def __init__(self, dim, heads, depth, layer_dropout=0.0):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=layer_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        def forward(self, x):
            return self.encoder(x)

    class Decoder(nn.Module):
        def __init__(self, dim, depth, heads):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=depth)

        def forward(self, x):
            return self.decoder(x)

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

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
except Exception:
    fallback_snntorch = types.ModuleType("snntorch")
    fallback_snntorch.Leaky = _StatelessSurrogateLeaky
    sys.modules["snntorch"] = fallback_snntorch
    import snntorch as snn


BETA = 0.9


class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)
        self.predict_lif = snn.Leaky(beta=BETA)

    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim=1)
        x = self.predictor(x)
        x, _ = self.predict_lif(x)
        return x[:, -target_masks.shape[1]:, :]


class RandomMaskIJEPA_fMRI_base(nn.Module):
    def __init__(
        self,
        num_regions=90,
        seq_len=240,
        embed_dim=64,
        enc_depth=8,
        pred_depth=6,
        num_heads=8,
        context_k=60,
        post_emb_norm=False,
        mode="train",
        layer_dropout=0.0,
    ):
        super().__init__()
        if not 0 < context_k < num_regions:
            raise ValueError("context_k must satisfy 0 < context_k < num_regions")

        self.num_regions = num_regions
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.context_k = context_k
        self.mode = mode
        self.layer_dropout = layer_dropout

        self.token_embed = nn.Linear(seq_len, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_regions, embed_dim))
        self.decoder_to_seq = nn.Linear(embed_dim, seq_len)

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.token_lif = snn.Leaky(beta=BETA)
        self.teacher_lif = snn.Leaky(beta=BETA)
        self.student_lif = snn.Leaky(beta=BETA)
        self.mask_lif = snn.Leaky(beta=BETA)
        self.output_lif = snn.Leaky(beta=BETA)

        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )
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

    def sample_mask_indices(self, device, context_k=None):
        if context_k is None:
            context_k = self.context_k
        if not 0 < context_k < self.num_regions:
            raise ValueError("context_k must satisfy 0 < context_k < num_regions")

        perm = torch.randperm(self.num_regions, device=device)
        context_indices = perm[:context_k].sort().values
        target_indices = perm[context_k:].sort().values
        return context_indices, target_indices

    def forward(self, x, context_k=None):
        B = x.shape[0]
        context_indices, target_indices = self.sample_mask_indices(x.device, context_k)

        x_emb = self.token_embed(x)
        x_emb = x_emb + self.pos_embedding
        x_emb = self.post_emb_norm(x_emb)
        x_emb, _ = self.token_lif(x_emb)

        if self.mode == "test":
            student_test = self.student_encoder(x_emb)
            student_test = self.norm(student_test)
            student_test, _ = self.student_lif(student_test)
            return student_test

        with torch.no_grad():
            teacher_full = self.get_teacher_full(x_emb)

        student_full = self.student_encoder(x_emb)
        student_full = self.norm(student_full)
        student_full, _ = self.student_lif(student_full)

        context_encoding = student_full[:, context_indices, :]
        teacher_targets = teacher_full[:, target_indices, :]

        target_pos = self.pos_embedding[:, target_indices, :]
        target_masks = self.mask_token.repeat(B, target_indices.numel(), 1) + target_pos
        target_masks, _ = self.mask_lif(target_masks)

        pred_targets = self.predictor(context_encoding, target_masks)

        pred_seq_current = self.decoder_to_seq(pred_targets)
        teacher_seq_current = self.decoder_to_seq(teacher_targets)

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
            "context_indices": context_indices.detach().cpu().tolist(),
            "target_indices": target_indices.detach().cpu().tolist(),
        }


def firing_rate_mse_loss(pred_rate, target_rate):
    return torch.mean((pred_rate - target_rate) ** 2)


_LightningModuleBase = pl.LightningModule if pl is not None else nn.Module
_LightningDataModuleBase = pl.LightningDataModule if pl is not None else object


class RandomMaskIJEPA_fMRI(_LightningModuleBase):
    def __init__(
        self,
        dataset_path=None,
        num_regions=90,
        seq_len=240,
        embed_dim=64,
        enc_heads=8,
        enc_depth=8,
        decoder_depth=6,
        context_k=60,
        lr=1e-4,
        weight_decay=0.05,
        m=0.996,
        m_start_end=(0.996, 1.0),
    ):
        super().__init__()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()
        self.model = RandomMaskIJEPA_fMRI_base(
            num_regions=num_regions,
            seq_len=seq_len,
            embed_dim=embed_dim,
            enc_depth=enc_depth,
            pred_depth=decoder_depth,
            num_heads=enc_heads,
            context_k=context_k,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.context_k = context_k
        self.criterion = firing_rate_mse_loss
        self.m = m
        self.m_start_end = m_start_end
        self.dataset_path = dataset_path

        self.val_sum_sq_error = None
        self.val_n_elements = None
        self.val_sum_targets = None
        self.val_sum_targets_sq = None

    def forward(self, x):
        return self.model(x, context_k=self.context_k)

    def update_momentum(self, m):
        student_model = self.model.student_encoder
        teacher_model = self.model.teacher_encoder
        with torch.no_grad():
            for s_p, t_p in zip(student_model.parameters(), teacher_model.parameters()):
                t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs["pred_rate"], outputs["target_rate"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_rate_mse", loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_sum_sq_error = 0.0
        self.val_n_elements = 0
        self.val_sum_targets = 0.0
        self.val_sum_targets_sq = 0.0

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        pred_rate = outputs["pred_rate"]
        target_rate = outputs["target_rate"]
        loss = self.criterion(pred_rate, target_rate)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        pred_cpu = pred_rate.detach().cpu()
        target_cpu = target_rate.detach().cpu()
        diff = pred_cpu - target_cpu

        self.val_sum_sq_error += (diff ** 2).sum().item()
        self.val_n_elements += pred_cpu.numel()
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
        rse = float("inf") if ss_tot <= 0 else rmse / (math.sqrt(ss_tot / float(self.val_n_elements)) + 1e-12)

        self.log("val_MSE", mse, prog_bar=True, on_epoch=True)
        self.log("val_RMSE", rmse, on_epoch=True)
        self.log("val_RSE", rse, prog_bar=True, on_epoch=True)
        self.log("val_R2", r2, prog_bar=True, on_epoch=True)
        self.print(f"VAL METRICS -- MSE: {mse:.6f}, RMSE: {rmse:.6f}, RSE: {rse:.6f}, R2: {r2:.6f}")

    def on_after_backward(self):
        self.update_momentum(self.m)
        if self.trainer is not None:
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            self.m += (self.m_start_end[1] - self.m_start_end[0]) / float(total_steps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.trainer is None:
            return optimizer

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


class FMRI_Dataset(Dataset):
    def __init__(self, mat_path, spike_model=None, device="cpu"):
        mat = sio.loadmat(mat_path)
        arr = mat["Data"]
        self.Data = arr.astype(np.float32)
        self.spike = spike_model
        self.device = device

        if self.spike is not None:
            self.spike = self.spike.to(self.device)
            self.spike.eval()

    def __len__(self):
        return self.Data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.Data[idx]).float()
        x = x.unsqueeze(0).to(self.device)
        if self.spike is not None:
            with torch.no_grad():
                x = self.spike(x)
        return x.squeeze(0).cpu()


class FMRIDataModule(_LightningDataModuleBase):
    def __init__(self, dataset_path, batch_size=8, num_workers=2, shuffle=True, device="cpu"):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.device = device

    def setup(self, stage=None):
        try:
            from SNN_fMRI import SNN_Model
        except ModuleNotFoundError as exc:
            if exc.name != "snn_layers":
                raise
            import SNN_layers as snn_layers

            sys.modules.setdefault("snn_layers", snn_layers)
            from SNN_fMRI import SNN_Model

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
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def run_smoke_test(args):
    torch.manual_seed(args.seed)
    model = RandomMaskIJEPA_fMRI_base(
        num_regions=args.num_regions,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        pred_depth=args.decoder_depth,
        num_heads=args.enc_heads,
        context_k=args.context_k,
    )
    x = torch.rand(args.batch_size, args.num_regions, args.seq_len)
    outputs = model(x)
    print("context_count:", len(outputs["context_indices"]))
    print("target_count:", len(outputs["target_indices"]))
    print("pred_seq_shape:", tuple(outputs["pred_seq"].shape))
    print("target_seq_shape:", tuple(outputs["target_seq"].shape))


def train(args):
    if pl is None:
        raise ModuleNotFoundError("pytorch_lightning is required for training. Install it or use --smoke-test.")

    from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary

    dm = FMRIDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    model = RandomMaskIJEPA_fMRI(
        dataset_path=args.dataset_path,
        num_regions=args.num_regions,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        enc_heads=args.enc_heads,
        enc_depth=args.enc_depth,
        decoder_depth=args.decoder_depth,
        context_k=args.context_k,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        max_epochs=args.max_epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=2),
        ],
        gradient_clip_val=0.1,
    )
    trainer.fit(model, dm)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Random-mask SpikeDB for fMRI region prediction.")
    parser.add_argument("--dataset-path", default=os.path.join("fMRI_Data", "test.mat"))
    parser.add_argument("--context-k", type=int, default=60, help="Number of visible context regions.")
    parser.add_argument("--num-regions", type=int, default=90)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Run one random forward pass and exit.")
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    if cli_args.smoke_test:
        run_smoke_test(cli_args)
    else:
        train(cli_args)
