import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from x_transformers import Encoder

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

        self.encoder = Encoder(dim=embed_dim, heads=num_heads, depth=enc_depth, layer_dropout=0.0)

        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, num_regions, seq_len]
        """
        x = self.token_embed(x) + self.pos_embedding     # [B, R, D]
        x = self.post_emb_norm(x)                        # [B, R, D]
        x = self.encoder(x)                              # [B, R, D]
        x = self.head_norm(x)                            # [B, R, D]  <-- LN 
        x = x.mean(dim=1)                                # [B, D]
        logits = self.head(x)                            # [B, num_classes]
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
    def __init__(self, data_path, label_path, zscore=False):
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

        assert len(X) == len(y), f"数据和标签数量不匹配: X={len(X)}, y={len(y)}"

        if zscore:
            mean = X.mean(axis=2, keepdims=True)
            std = X.std(axis=2, keepdims=True) + 1e-6
            X = (X - mean) / std

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])           # [R, T]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class FMRIDataModule(pl.LightningDataModule):
    def __init__(self, data_path, label_path, batch_size=8, num_workers=2, shuffle=True, zscore=False):
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.zscore = zscore

    def setup(self, stage=None):
        full_dataset = FMRI_Classification_Dataset(self.data_path, self.label_path, zscore=self.zscore)
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

    dm = FMRIDataModule(data_path, label_path, batch_size=2, num_workers=2, shuffle=True, zscore=False)

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
