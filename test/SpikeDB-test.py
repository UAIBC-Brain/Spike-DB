import argparse
import glob
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split
from SpikeDB import IJEPA_fMRI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


DEFAULT_DATASET = r"XX.mat"


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


def build_test_dataset(dataset_path, device, use_full_data, seed):
    from SpikeDB import FMRI_Dataset
    from SNN_fMRI import SNN_Model

    spike_model = SNN_Model()
    full_dataset = FMRI_Dataset(dataset_path, spike_model=spike_model, device=device)

    if use_full_data:
        return full_dataset

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    return test_dataset


def load_model(args, checkpoint_path, device):


    if checkpoint_path is None:
        if not args.allow_random_init:
            raise FileNotFoundError(
                "No valid checkpoint was found. Pass --checkpoint path/to/model.ckpt "
                "or use --allow-random-init to test an untrained model."
            )

        print("WARNING: no checkpoint loaded; evaluating a randomly initialized model.")
        return IJEPA_fMRI(
            dataset_path=args.dataset,
            num_regions=args.num_regions,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            enc_heads=args.enc_heads,
            enc_depth=args.enc_depth,
            decoder_depth=args.decoder_depth,
            M=args.M,
            exhaustive=args.exhaustive,
        ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = IJEPA_fMRI.load_from_checkpoint(
        checkpoint_path,
        dataset_path=args.dataset,
        map_location=device,
    )
    return model.to(device)


def evaluate(model, dataloader, device):
    model.eval()

    sum_sq_error = 0.0
    sum_abs_error = 0.0
    n_elements = 0
    sum_targets = 0.0
    sum_targets_sq = 0.0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            x = batch.to(device)
            outputs = model(x)

            pred_rate = outputs["pred_rate"].detach().cpu()
            target_rate = outputs["target_rate"].detach().cpu()

            diff = pred_rate - target_rate
            sum_sq_error += (diff ** 2).sum().item()
            sum_abs_error += diff.abs().sum().item()
            n_elements += pred_rate.numel()
            sum_targets += target_rate.sum().item()
            sum_targets_sq += (target_rate ** 2).sum().item()

            if batch_idx == 0:
                print(f"Prediction shape: {tuple(pred_rate.shape)}")
                print(f"Target shape:     {tuple(target_rate.shape)}")

    if n_elements == 0:
        raise RuntimeError("The test dataloader is empty.")

    mse = sum_sq_error / float(n_elements)
    rmse = math.sqrt(mse)
    mae = sum_abs_error / float(n_elements)
    mean_targets = sum_targets / float(n_elements)
    ss_tot = sum_targets_sq - float(n_elements) * (mean_targets ** 2)
    r2 = 0.0 if ss_tot <= 0 else 1.0 - (sum_sq_error / ss_tot)

    if ss_tot <= 0:
        rse = float("inf")
    else:
        std_y = math.sqrt(ss_tot / float(n_elements))
        rse = rmse / (std_y + 1e-12)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "RSE": rse,
        "R2": r2,
        "N": n_elements,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SpikeDB checkpoint.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to the .mat fMRI file.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained .ckpt file.")
    parser.add_argument(
        "--log-dir",
        default=os.path.join(SCRIPT_DIR, "lightning_logs"),
        help="Directory searched when --checkpoint is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-data", action="store_true", help="Evaluate on the whole dataset.")
    parser.add_argument("--allow-random-init", action="store_true")

    parser.add_argument("--num-regions", type=int, default=90)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--M", type=int, default=90)
    parser.add_argument("--exhaustive", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.log_dir)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print("Split: full dataset" if args.full_data else "Split: held-out 20% test split")

    test_dataset = build_test_dataset(
        dataset_path=args.dataset,
        device=device,
        use_full_data=args.full_data,
        seed=args.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = load_model(args, checkpoint_path, device)
    metrics = evaluate(model, test_loader, device)

    print("\nTEST METRICS")
    for name in ["MSE", "RMSE", "MAE", "RSE", "R2"]:
        print(f"{name}: {metrics[name]:.6f}")
    print(f"N: {metrics['N']}")


if __name__ == "__main__":
    main()
