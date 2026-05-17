import argparse
import glob
import math
import os
import sys

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset, random_split


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


DEFAULT_DATASET = os.path.join(PROJECT_DIR, "fMRI_Data", "test.mat")


class RawFMRIDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

        file_size = os.path.getsize(dataset_path)
        if file_size < 128:
            raise ValueError(
                f"Dataset file is too small to be a valid .mat file: {dataset_path} "
                f"({file_size} bytes). Pass --dataset path/to/your_real_data.mat."
            )

        mat = sio.loadmat(dataset_path)
        if "Data" not in mat:
            raise KeyError(f"Could not find key 'Data' in {dataset_path}.")
        self.data = mat["Data"].astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()


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


def build_test_dataset(dataset_path, use_full_data, seed):
    full_dataset = RawFMRIDataset(dataset_path)
    if use_full_data:
        return full_dataset

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    if test_size <= 0:
        raise ValueError(f"Test split is empty. Dataset length={len(full_dataset)}.")

    generator = torch.Generator().manual_seed(seed)
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    return test_dataset


def build_base_model(args):
    from RandomMaskSpikeDB import RandomMaskIJEPA_fMRI_base

    return RandomMaskIJEPA_fMRI_base(
        num_regions=args.num_regions,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        pred_depth=args.decoder_depth,
        num_heads=args.enc_heads,
        context_k=args.context_k,
    )


def strip_known_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_model(args, checkpoint_path, device):
    if checkpoint_path is None:
        if not args.allow_random_init:
            raise FileNotFoundError(
                "No valid checkpoint was found. Pass --checkpoint path/to/model.ckpt "
                "or use --allow-random-init to test an untrained model."
            )
        print("WARNING: no checkpoint loaded; evaluating a randomly initialized model.")
        return build_base_model(args).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_base_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = strip_known_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"WARNING: missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"WARNING: unexpected keys while loading checkpoint: {len(unexpected)}")

    return model.to(device)


def evaluate(model, dataloader, device):
    model.eval()

    sum_sq_error = 0.0
    sum_abs_error = 0.0
    n_elements = 0
    sum_targets = 0.0
    sum_targets_sq = 0.0

    first_context_indices = None
    first_target_indices = None

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
                first_context_indices = outputs["context_indices"]
                first_target_indices = outputs["target_indices"]
                print(f"Prediction rate shape: {tuple(pred_rate.shape)}")
                print(f"Target rate shape:     {tuple(target_rate.shape)}")
                print(f"Context count: {len(first_context_indices)}")
                print(f"Target count:  {len(first_target_indices)}")
                print(f"Context indices: {first_context_indices}")
                print(f"Target indices:  {first_target_indices}")

    if n_elements == 0:
        raise RuntimeError("The test dataloader is empty.")

    mse = sum_sq_error / float(n_elements)
    rmse = math.sqrt(mse)
    mae = sum_abs_error / float(n_elements)
    mean_targets = sum_targets / float(n_elements)
    ss_tot = sum_targets_sq - float(n_elements) * (mean_targets ** 2)
    r2 = 0.0 if ss_tot <= 0 else 1.0 - (sum_sq_error / ss_tot)
    rse = float("inf") if ss_tot <= 0 else rmse / (math.sqrt(ss_tot / float(n_elements)) + 1e-12)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "RSE": rse,
        "R2": r2,
        "N": n_elements,
    }


def run_smoke_test(args, device):
    torch.manual_seed(args.seed)
    model = build_base_model(args).to(device)
    model.eval()
    x = torch.rand(args.batch_size, args.num_regions, args.seq_len, device=device)

    with torch.inference_mode():
        outputs = model(x)

    print("SMOKE TEST")
    print(f"Context count: {len(outputs['context_indices'])}")
    print(f"Target count:  {len(outputs['target_indices'])}")
    print(f"Pred seq shape:   {tuple(outputs['pred_seq'].shape)}")
    print(f"Target seq shape: {tuple(outputs['target_seq'].shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RandomMaskSpikeDB with random target-region masks.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to the .mat fMRI file with key 'Data'.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained .ckpt file.")
    parser.add_argument(
        "--log-dir",
        default=os.path.join(PROJECT_DIR, "lightning_logs"),
        help="Directory searched when --checkpoint is not provided.",
    )
    parser.add_argument("--context-k", type=int, default=60, help="Number of visible context regions.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-data", action="store_true", help="Evaluate on the whole dataset.")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use random input and skip dataset/checkpoint loading.")

    parser.add_argument("--num-regions", type=int, default=90)
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--decoder-depth", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Context K: {args.context_k}")
    print(f"Target regions: {args.num_regions - args.context_k}")

    if args.smoke_test:
        run_smoke_test(args, device)
        return

    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.log_dir)
    print(f"Dataset: {args.dataset}")
    print("Split: full dataset" if args.full_data else "Split: held-out 20% test split")

    test_dataset = build_test_dataset(
        dataset_path=args.dataset,
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
