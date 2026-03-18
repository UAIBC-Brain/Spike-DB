import torch
import scipy.io as sio
import numpy as np
import os
from SpikeDB import IJEPA_fMRI   

ckpt_path = r"G:\XXX\XXX.ckpt"
data_path = r"G:\XXX\XXX.mat"
sample_idx = 50          
exclude_self = True       
device = "cuda" if torch.cuda.is_available() else "cpu"

perturbation_method = "gaussian_bump"   
perturb_params = {
    "add_constant": dict(k_std=0.5, value=None),
    "scale": dict(alpha=0.5),
    "impulse": dict(t0=80, t1=120, k_std=1.0, value=None),
    "gaussian_bump": dict(tc=95, sigma=20, k_std=1.0, value=None),   
    "sinusoid": dict(freq=2.0, phi=0.0, k_std=1.0, value=None),
    "noise": dict(k_std=0.5),
    "silence": dict(mode="mean"),
}

scale_factor = 10000.0
threshold = 0.001   

mat = sio.loadmat(data_path)

if "Data" in mat:
    Data = mat["Data"]
else:
    Data = None
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            Data = v
            break
    if Data is None:
        raise KeyError(f"无法在 {data_path} 中找到变量 'Data' 或其他三维数组。")

Data = Data.astype(np.float32)  # [num_samples, num_regions, seq_len]
num_samples, num_regions, seq_len = Data.shape

if not (0 <= sample_idx < num_samples):
    raise ValueError(f"样本索引超出范围: 0 <= sample_idx < {num_samples}")

x_orig = Data[sample_idx]                   # [R, T]
x_tensor = torch.from_numpy(x_orig).unsqueeze(0).to(device)  # [1, R, T]


try:
    model = IJEPA_fMRI.load_from_checkpoint(ckpt_path, dataset_path=data_path)
except TypeError:
    model = IJEPA_fMRI.load_from_checkpoint(ckpt_path)
model.eval().to(device)


with torch.no_grad():
    out = model(x_tensor)
    if isinstance(out, (tuple, list)):
        pred_orig = out[0]
    else:
        pred_orig = out
pred_orig = pred_orig.squeeze(0).cpu().numpy()  # [R, T]


def apply_perturbation(x_batch, src_region, method, params, seq_len):
    x_p = x_batch.clone()
    x_src = x_p[0, src_region, :]  # [T]
    src_mean = x_src.mean()
    src_std = x_src.std().clamp(min=1e-12)

    def _resolve_amp(k_std=None, value=None):
        if value is not None:
            return torch.tensor(float(value), dtype=x_src.dtype, device=x_src.device)
        k = 0.0 if k_std is None else float(k_std)
        return src_std * k

    if method == "add_constant":
        amp = _resolve_amp(**params.get("add_constant", {}))
        x_src = x_src + amp

    elif method == "scale":
        alpha = float(params.get("scale", {}).get("alpha", 0.2))
        x_src = x_src * (1.0 + alpha)

    elif method == "impulse":
        cfg = params.get("impulse", {})
        t0 = int(cfg.get("t0", 0))
        t1 = int(cfg.get("t1", min(30, seq_len)))
        t0 = max(0, min(t0, seq_len))
        t1 = max(t0+1, min(t1, seq_len))
        amp = _resolve_amp(k_std=cfg.get("k_std", 1.0), value=cfg.get("value", None))
        mask = torch.zeros_like(x_src)
        mask[t0:t1] = 1.0
        x_src = x_src + amp * mask

    elif method == "gaussian_bump":
        cfg = params.get("gaussian_bump", {})
        tc = float(cfg.get("tc", seq_len/2))
        sigma = float(cfg.get("sigma", 20.0))
        amp = _resolve_amp(k_std=cfg.get("k_std", 1.0), value=cfg.get("value", None))
        t = torch.arange(seq_len, device=x_src.device, dtype=x_src.dtype)
        bump = torch.exp(-0.5 * ((t - tc) / max(1e-6, sigma))**2)
        bump = bump / (bump.max().clamp(min=1e-12))
        x_src = x_src + amp * bump

    elif method == "sinusoid":
        cfg = params.get("sinusoid", {})
        freq = float(cfg.get("freq", 2.0))
        phi = float(cfg.get("phi", 0.0))
        amp = _resolve_amp(k_std=cfg.get("k_std", 1.0), value=cfg.get("value", None))
        t = torch.arange(seq_len, device=x_src.device, dtype=x_src.dtype)
        sinus = torch.sin(2.0 * np.pi * freq * t / seq_len + phi)
        sinus = sinus.to(x_src.dtype)
        x_src = x_src + amp * sinus

    elif method == "noise":
        cfg = params.get("noise", {})
        k_std = float(cfg.get("k_std", 0.5))
        noise = torch.randn_like(x_src) * src_std * k_std
        x_src = x_src + noise

    elif method == "silence":
        mode = params.get("silence", {}).get("mode", "mean")
        if mode == "zero":
            x_src = torch.zeros_like(x_src)
        else:
            x_src = torch.ones_like(x_src) * src_mean

    else:
        raise ValueError(f"未知扰动方法: {method}")

    x_p[0, src_region, :] = x_src
    return x_p


EC_matrix = np.zeros((num_regions, num_regions), dtype=np.float64)
per_source_best = []  # (src, best_tgt, best_signed_val, pos_idx, pos_val, neg_idx, neg_val)

print("\n开始扰动并计算（差异已乘以 {}，阈值 = {}）...\n".format(int(scale_factor), threshold))

for src_region in range(num_regions):
    x_perturbed = apply_perturbation(x_tensor, src_region, perturbation_method, perturb_params, seq_len)
    with torch.no_grad():
        out_new = model(x_perturbed)
        if isinstance(out_new, (tuple, list)):
            pred_new = out_new[0]
        else:
            pred_new = out_new
    pred_new = pred_new.squeeze(0).cpu().numpy()  # [R, T]

    delta = pred_new - pred_orig                   # [R, T]
    delta_mean = delta.mean(axis=1) * scale_factor # [R]

    if exclude_self:
        delta_mean[src_region] = 0.0

    EC_matrix[src_region, :] = delta_mean

    abs_idx = int(np.argmax(np.abs(delta_mean)))
    abs_signed_val = float(delta_mean[abs_idx])

    pos_vals = delta_mean.copy(); pos_vals[pos_vals <= 0] = -np.inf
    if np.isfinite(pos_vals).any():
        pos_idx = int(np.argmax(pos_vals)); pos_val = float(delta_mean[pos_idx])
    else:
        pos_idx, pos_val = None, None

    neg_vals = delta_mean.copy(); neg_vals[neg_vals >= 0] = np.inf
    if np.isfinite(neg_vals).any():
        neg_idx = int(np.argmin(neg_vals)); neg_val = float(delta_mean[neg_idx])
    else:
        neg_idx, neg_val = None, None

    per_source_best.append((src_region, abs_idx, abs_signed_val, pos_idx, pos_val, neg_idx, neg_val))

    if abs(abs_signed_val) > threshold / 10000:
        argmax_tgt = int(np.argmax(delta_mean))
        argmin_tgt = int(np.argmin(delta_mean))
        kind = "兴奋性" if abs_signed_val > 0 else "抑制性" if abs_signed_val < 0 else "中性"
        print(
            f"扰动源脑区 {src_region}: "
            f"argmax 目标={argmax_tgt} (值={delta_mean[argmax_tgt]:.8f}), "
            f"argmin 目标={argmin_tgt} (值={delta_mean[argmin_tgt]:.8f}), "
            f"|.|最大 目标={abs_idx} (值={abs_signed_val:.8f}, 类型={kind})"
        )

global_list = [(src, tgt, signed_val) for (src, tgt, signed_val, *_ ) in per_source_best]

global_filtered_sorted = sorted([x for x in global_list if abs(x[2]) > threshold],
                                key=lambda y: abs(y[2]), reverse=True)

if len(global_filtered_sorted) > 0:
    print("\n基于‘每个源脑区最突出连接’（按绝对值降序，已过滤 |val| > {}）:\n".format(threshold))
    for src, tgt, val in global_filtered_sorted:
        kind = "兴奋性" if val > 0 else "抑制性" if val < 0 else "中性"
        print(f"源脑区 {src} -> 目标脑区 {tgt} : {val:.8f} ({kind})")
    print(f"\n共 {len(global_filtered_sorted)} 条符合条件的全局连接（无 TopN 限制）。")
else:
    print(f"\n未找到任何 |val| > {threshold} 的全局突出连接。")


