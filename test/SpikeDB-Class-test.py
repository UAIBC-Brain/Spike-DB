import argparse
import glob
import importlib.util
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
CLASS_FILE = os.path.join(PROJECT_DIR, "SpikeDB-Class.py")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def load_spikedb_class_module():
    spec = importlib.util.spec_from_file_location("spikedb_class", CLASS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {CLASS_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("spikedb_class", module)
    spec.loader.exec_module(module)
    return module


def find_latest_checkpoint(log_dir):
    pattern = os.path.join(log_dir, "**", "*.ckpt")
    candidates = [
        path for path in glob.glob(pattern, recursive=True)
        if os.path.isfile(path) and os.path.getsize(path) > 1024
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_test_dataset(spikedb_class, args, device):
    snn_model = spikedb_class.SNN_Model()
    full_dataset = spikedb_class.FMRI_Classification_Dataset(
        data_path=args.data,
        label_path=args.label,
        zscore=args.zscore,
        spike_model=snn_model,
        device=device,
    )

    if args.full_data:
        return full_dataset

    train_size = int(args.train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    if test_size <= 0:
        raise ValueError(
            f"Test split is empty. Dataset length={len(full_dataset)}, "
            f"train_ratio={args.train_ratio}."
        )

    generator = torch.Generator().manual_seed(args.seed)
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    return test_dataset


def load_model(spikedb_class, args, checkpoint_path, device):
    if checkpoint_path is None:
        if not args.allow_random_init:
            raise FileNotFoundError(
                "No valid checkpoint was found. Pass --checkpoint path/to/model.ckpt "
                "or use --allow-random-init to evaluate an untrained model."
            )

        print("WARNING: no checkpoint loaded; evaluating a randomly initialized model.")
        return spikedb_class.IJEPA_fMRI_Lit(
            dataset_path=args.data,
            label_path=args.label,
            num_regions=args.num_regions,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            enc_heads=args.enc_heads,
            enc_depth=args.enc_depth,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_classes=args.num_classes,
            dropout=args.dropout,
        ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = spikedb_class.IJEPA_fMRI_Lit.load_from_checkpoint(
        checkpoint_path,
        dataset_path=args.data,
        label_path=args.label,
        map_location=device,
    )
    return model.to(device)


def compute_metrics(preds, labels, num_classes):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    acc = correct / total if total else 0.0

    metrics = {"ACC": acc}
    if num_classes == 2:
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        metrics["SEN"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["SPE"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["F1"] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        metrics["TP"] = tp
        metrics["TN"] = tn
        metrics["FP"] = fp
        metrics["FN"] = fn

    return metrics


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_samples += y.numel()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            if batch_idx == 0:
                print(f"Logits shape: {tuple(logits.shape)}")
                print(f"Labels shape: {tuple(y.shape)}")

    if total_samples == 0:
        raise RuntimeError("The test dataloader is empty.")

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(preds, labels, num_classes)
    metrics["loss"] = total_loss / float(total_samples)
    metrics["N"] = total_samples
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SpikeDB-Class checkpoint.")
    parser.add_argument("--data", required=True, help="Path to the .mat fMRI data file with key 'Data'.")
    parser.add_argument("--label", required=True, help="Path to the .mat label file with key 'Y'.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained .ckpt file.")
    parser.add_argument(
        "--log-dir",
        default=os.path.join(PROJECT_DIR, "lightning_logs"),
        help="Directory searched when --checkpoint is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--full-data", action="store_true", help="Evaluate on the whole dataset.")
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")

    parser.add_argument("--num-regions", type=int, default=90)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.log_dir)

    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Label: {args.label}")
    print("Split: full dataset" if args.full_data else "Split: held-out test split")

    spikedb_class = load_spikedb_class_module()
    test_dataset = build_test_dataset(spikedb_class, args, device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = load_model(spikedb_class, args, checkpoint_path, device)
    metrics = evaluate(model, test_loader, device, args.num_classes)

    print("\nTEST METRICS")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"ACC: {metrics['ACC']:.6f}")
    if args.num_classes == 2:
        print(f"SEN: {metrics['SEN']:.6f}")
        print(f"SPE: {metrics['SPE']:.6f}")
        print(f"F1:  {metrics['F1']:.6f}")
        print(
            "Confusion matrix counts: "
            f"TP={metrics['TP']}, TN={metrics['TN']}, "
            f"FP={metrics['FP']}, FN={metrics['FN']}"
        )
    print(f"N: {metrics['N']}")


if __name__ == "__main__":
    main()
