import argparse
import glob
import os
import sys

import numpy as np
import scipy.io as sio
import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


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


def load_data(data_path):
    mat = sio.loadmat(data_path)
    if "Data" in mat:
        data = mat["Data"]
    else:
        data = None
        for value in mat.values():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                data = value
                break
        if data is None:
            raise KeyError(f"Could not find 'Data' or any other 3D array in {data_path}.")

    return data.astype(np.float32)


def load_model(checkpoint_path, data_path, device):
    from SpikeDB import IJEPA_fMRI

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        model = IJEPA_fMRI.load_from_checkpoint(
            checkpoint_path,
            dataset_path=data_path,
            map_location=device,
        )
    except TypeError:
        model = IJEPA_fMRI.load_from_checkpoint(checkpoint_path, map_location=device)

    model.eval().to(device)
    return model


def extract_prediction_sequence(model_output):
    if isinstance(model_output, dict):
        if "pred_seq" in model_output:
            return model_output["pred_seq"]
        if "target_seq" in model_output:
            return model_output["target_seq"]
        raise KeyError("Model output dict does not contain 'pred_seq' or 'target_seq'.")

    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    return model_output


def resolve_amp(x_src, k_std=None, value=None):
    if value is not None:
        return torch.tensor(float(value), dtype=x_src.dtype, device=x_src.device)

    k = 0.0 if k_std is None else float(k_std)
    src_std = x_src.std().clamp(min=1e-12)
    return src_std * k


def apply_perturbation(x_batch, src_region, method, args, seq_len):
    x_p = x_batch.clone()
    x_src = x_p[0, src_region, :]
    src_mean = x_src.mean()

    if method == "add_constant":
        x_src = x_src + resolve_amp(x_src, k_std=args.k_std, value=args.value)

    elif method == "scale":
        x_src = x_src * (1.0 + args.alpha)

    elif method == "impulse":
        t0 = max(0, min(args.t0, seq_len))
        t1 = max(t0 + 1, min(args.t1, seq_len))
        mask = torch.zeros_like(x_src)
        mask[t0:t1] = 1.0
        x_src = x_src + resolve_amp(x_src, k_std=args.k_std, value=args.value) * mask

    elif method == "gaussian_bump":
        t = torch.arange(seq_len, device=x_src.device, dtype=x_src.dtype)
        sigma = max(1e-6, args.sigma)
        bump = torch.exp(-0.5 * ((t - args.tc) / sigma) ** 2)
        bump = bump / bump.max().clamp(min=1e-12)
        x_src = x_src + resolve_amp(x_src, k_std=args.k_std, value=args.value) * bump

    elif method == "sinusoid":
        t = torch.arange(seq_len, device=x_src.device, dtype=x_src.dtype)
        sinus = torch.sin(2.0 * np.pi * args.freq * t / seq_len + args.phi)
        x_src = x_src + resolve_amp(x_src, k_std=args.k_std, value=args.value) * sinus

    elif method == "noise":
        src_std = x_src.std().clamp(min=1e-12)
        x_src = x_src + torch.randn_like(x_src) * src_std * args.k_std

    elif method == "silence":
        if args.silence_mode == "zero":
            x_src = torch.zeros_like(x_src)
        else:
            x_src = torch.ones_like(x_src) * src_mean

    else:
        raise ValueError(f"Unknown perturbation method: {method}")

    x_p[0, src_region, :] = x_src
    return x_p


def run_ec_analysis(model, x_tensor, args, num_regions, seq_len):
    with torch.inference_mode():
        out = model(x_tensor)
        pred_orig = extract_prediction_sequence(out)

    pred_orig = pred_orig.squeeze(0).detach().cpu().numpy()
    if pred_orig.ndim != 2:
        raise ValueError(f"Expected prediction sequence shape [R, T], got {pred_orig.shape}.")

    ec_matrix = np.zeros((num_regions, num_regions), dtype=np.float64)
    per_source_best = []

    print(
        "\nStarting perturbation analysis "
        f"(method={args.method}, scale_factor={args.scale_factor}, threshold={args.threshold})...\n"
    )

    for src_region in range(num_regions):
        x_perturbed = apply_perturbation(x_tensor, src_region, args.method, args, seq_len)
        with torch.inference_mode():
            out_new = model(x_perturbed)
            pred_new = extract_prediction_sequence(out_new)

        pred_new = pred_new.squeeze(0).detach().cpu().numpy()
        delta = pred_new - pred_orig
        delta_mean = delta.mean(axis=1) * args.scale_factor

        if args.exclude_self:
            delta_mean[src_region] = 0.0

        ec_matrix[src_region, :] = delta_mean
        abs_idx = int(np.argmax(np.abs(delta_mean)))
        abs_signed_val = float(delta_mean[abs_idx])

        per_source_best.append((src_region, abs_idx, abs_signed_val))

        if abs(abs_signed_val) > args.print_threshold:
            argmax_tgt = int(np.argmax(delta_mean))
            argmin_tgt = int(np.argmin(delta_mean))
            kind = "excitatory" if abs_signed_val > 0 else "inhibitory" if abs_signed_val < 0 else "neutral"
            print(
                f"Source region {src_region}: "
                f"argmax target={argmax_tgt} (value={delta_mean[argmax_tgt]:.8f}), "
                f"argmin target={argmin_tgt} (value={delta_mean[argmin_tgt]:.8f}), "
                f"largest |effect| target={abs_idx} "
                f"(value={abs_signed_val:.8f}, type={kind})"
            )

    return ec_matrix, per_source_best


def save_outputs(ec_matrix, per_source_best, args):
    os.makedirs(args.output_dir, exist_ok=True)

    matrix_npy = os.path.join(args.output_dir, args.output_prefix + "_ec_matrix.npy")
    matrix_csv = os.path.join(args.output_dir, args.output_prefix + "_ec_matrix.csv")
    best_csv = os.path.join(args.output_dir, args.output_prefix + "_best_links.csv")

    np.save(matrix_npy, ec_matrix)
    np.savetxt(matrix_csv, ec_matrix, delimiter=",", fmt="%.10f")

    with open(best_csv, "w", encoding="utf-8") as f:
        f.write("source_region,target_region,value,type\n")
        for src, tgt, val in per_source_best:
            kind = "excitatory" if val > 0 else "inhibitory" if val < 0 else "neutral"
            f.write(f"{src},{tgt},{val:.10f},{kind}\n")

    print(f"\nSaved EC matrix: {matrix_npy}")
    print(f"Saved EC matrix CSV: {matrix_csv}")
    print(f"Saved strongest links CSV: {best_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpikeDB effective-connectivity perturbation test.")
    parser.add_argument("--data", required=True, help="Path to the .mat fMRI data file.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained SpikeDB .ckpt file.")
    parser.add_argument(
        "--log-dir",
        default=os.path.join(PROJECT_DIR, "lightning_logs"),
        help="Directory searched when --checkpoint is not provided.",
    )
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--method",
        default="gaussian_bump",
        choices=["add_constant", "scale", "impulse", "gaussian_bump", "sinusoid", "noise", "silence"],
    )
    parser.add_argument("--exclude-self", action="store_true", default=True)
    parser.add_argument("--include-self", dest="exclude_self", action="store_false")
    parser.add_argument("--scale-factor", type=float, default=10000.0)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--print-threshold", type=float, default=0.0000001)

    parser.add_argument("--k-std", type=float, default=1.0)
    parser.add_argument("--value", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t0", type=int, default=80)
    parser.add_argument("--t1", type=int, default=120)
    parser.add_argument("--tc", type=float, default=95.0)
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--freq", type=float, default=2.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--silence-mode", default="mean", choices=["mean", "zero"])

    parser.add_argument("--output-dir", default=TEST_DIR)
    parser.add_argument("--output-prefix", default="SpikeDB-EC-test")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.log_dir)
    if checkpoint_path is None:
        raise FileNotFoundError("No valid checkpoint found. Please pass --checkpoint path/to/model.ckpt.")

    data = load_data(args.data)
    num_samples, num_regions, seq_len = data.shape
    if not (0 <= args.sample_idx < num_samples):
        raise ValueError(f"Sample index out of range: 0 <= sample_idx < {num_samples}")

    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Data shape: {data.shape}")

    x_orig = data[args.sample_idx]
    x_tensor = torch.from_numpy(x_orig).unsqueeze(0).to(device)
    model = load_model(checkpoint_path, args.data, device)

    ec_matrix, per_source_best = run_ec_analysis(model, x_tensor, args, num_regions, seq_len)
    save_outputs(ec_matrix, per_source_best, args)

    global_filtered_sorted = sorted(
        [item for item in per_source_best if abs(item[2]) > args.threshold],
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    print(
        "\nGlobal strongest links from each source region "
        f"filtered by |value| > {args.threshold}:"
    )
    if not global_filtered_sorted:
        print("No links met the threshold.")
        return

    for src, tgt, val in global_filtered_sorted:
        kind = "excitatory" if val > 0 else "inhibitory" if val < 0 else "neutral"
        print(f"Source region {src} -> target region {tgt}: {val:.8f} ({kind})")


if __name__ == "__main__":
    main()
