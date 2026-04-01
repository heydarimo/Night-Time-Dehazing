import os
import re
import json
import csv
import argparse
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ssim

from model_flashinternimage_histoforme2 import fusion_net_histoformer2


# -----------------------------
# Optional metrics
# -----------------------------
LPIPS_AVAILABLE = True
SKVIDEO_AVAILABLE = True

try:
    import lpips
except Exception:
    LPIPS_AVAILABLE = False

try:
    from skvideo.measure import niqe as skvideo_niqe
except Exception:
    SKVIDEO_AVAILABLE = False


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG', '.webp')


def is_image_file(filename: str) -> bool:
    return filename.endswith(IMG_EXTENSIONS)


def extract_pair_key(filename: str) -> str:
    """
    Same pairing rule as training dataset.

    Examples:
      01_hazy.png -> 1
      01_GT.png   -> 1
      041.png     -> 41
      1400_10.png -> 1400
      1400.png    -> 1400
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^(\d+)', stem)
    if m is None:
        raise ValueError(f"Cannot extract pairing key from filename: {filename}")
    return str(int(m.group(1)))


def build_file_map(folder: str) -> dict:
    files = [f for f in os.listdir(folder) if is_image_file(f)]
    mapping = {}
    for f in files:
        key = extract_pair_key(f)
        if key in mapping:
            raise RuntimeError(f"Duplicate key '{key}' found in folder: {folder}")
        mapping[key] = f
    return mapping


class PairedTestDataset(Dataset):
    """
    Inference/evaluation dataset using the SAME pairing logic as PairedDehazeDataset.
    Expected structure:
        data_root/
            hazy/
            gt/
    """
    def __init__(self, data_root: str):
        super().__init__()
        self.transform = transforms.ToTensor()

        self.hazy_dir = os.path.join(data_root, 'hazy')
        self.gt_dir = os.path.join(data_root, 'gt')

        if not os.path.isdir(self.hazy_dir):
            raise FileNotFoundError(f"Hazy directory not found: {self.hazy_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        hazy_map = build_file_map(self.hazy_dir)
        gt_map = build_file_map(self.gt_dir)

        common_keys = sorted(set(hazy_map.keys()) & set(gt_map.keys()), key=lambda x: int(x))

        missing_in_gt = sorted(set(hazy_map.keys()) - set(gt_map.keys()), key=lambda x: int(x))
        missing_in_hazy = sorted(set(gt_map.keys()) - set(hazy_map.keys()), key=lambda x: int(x))

        if missing_in_gt:
            raise RuntimeError(f"Missing GT files for keys: {missing_in_gt}")
        if missing_in_hazy:
            raise RuntimeError(f"Missing hazy files for keys: {missing_in_hazy}")
        if len(common_keys) == 0:
            raise RuntimeError(f"No matched hazy/gt pairs found in {data_root}")

        self.samples = []
        for key in common_keys:
            hazy_path = os.path.join(self.hazy_dir, hazy_map[key])
            gt_path = os.path.join(self.gt_dir, gt_map[key])
            self.samples.append((hazy_path, gt_path, key, hazy_map[key], gt_map[key]))

        print(f"[PairedTestDataset] Loaded {len(self.samples)} pairs from {data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hazy_path, gt_path, key, hazy_name, gt_name = self.samples[index]

        hazy = Image.open(hazy_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if hazy.size != gt.size:
            raise RuntimeError(
                f"Size mismatch for key {key}: "
                f"hazy={hazy.size}, gt={gt.size}, "
                f"hazy_path={hazy_path}, gt_path={gt_path}"
            )

        hazy = self.transform(hazy)
        gt = self.transform(gt)

        return hazy, gt, key, hazy_name, gt_name


def safe_pad_2d(x, pad_left, pad_right, pad_top, pad_bottom, mode="reflect"):
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return x

    _, _, h, w = x.shape

    if mode == "reflect":
        if (w <= 1 and (pad_left + pad_right) > 0) or (h <= 1 and (pad_top + pad_bottom) > 0):
            return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        if pad_left > w - 1 or pad_right > w - 1 or pad_top > h - 1 or pad_bottom > h - 1:
            return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    if mode == "replicate":
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)


@torch.no_grad()
def forward_full_pad(model, x, mod=32, pad_mode="reflect"):
    b, c, H, W = x.shape
    assert b == 1

    H2 = ((H + mod - 1) // mod) * mod
    W2 = ((W + mod - 1) // mod) * mod

    pad_bottom = H2 - H
    pad_right = W2 - W

    xpad = safe_pad_2d(x, 0, pad_right, 0, pad_bottom, mode=pad_mode)
    pred = model(xpad)
    pred = pred[:, :, :H, :W]
    return pred


@torch.no_grad()
def forward_tile_simple(model, x, tile=384, overlap=96, pad_mode="reflect"):
    b, c, H, W = x.shape
    assert b == 1
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be greater than overlap")

    yy = torch.linspace(0, 1, tile, device=x.device)
    xx = torch.linspace(0, 1, tile, device=x.device)
    wy = (1.0 - (2.0 * (yy - 0.5)).abs()).clamp_min(0.0)
    wx = (1.0 - (2.0 * (xx - 0.5)).abs()).clamp_min(0.0)
    w2 = (wy[:, None] * wx[None, :])[None, None, :, :]

    out = torch.zeros((1, c, H, W), device=x.device, dtype=x.dtype)
    acc = torch.zeros((1, 1, H, W), device=x.device, dtype=x.dtype)

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + tile, H)
            right = min(left + tile, W)
            top0 = max(0, bottom - tile)
            left0 = max(0, right - tile)

            patch = x[:, :, top0:bottom, left0:right]
            ph, pw = patch.shape[-2], patch.shape[-1]

            pad_h = tile - ph
            pad_w = tile - pw
            if pad_h > 0 or pad_w > 0:
                patch = safe_pad_2d(patch, 0, pad_w, 0, pad_h, mode=pad_mode)

            pred = model(patch)[:, :, :ph, :pw]
            wcur = w2[:, :, :ph, :pw]

            out[:, :, top0:bottom, left0:right] += pred * wcur
            acc[:, :, top0:bottom, left0:right] += wcur

    out = out / acc.clamp_min(1e-6)
    return out


def tensor_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return x


def rgb_to_gray_uint8(x: torch.Tensor) -> np.ndarray:
    rgb = tensor_to_uint8_hwc(x).astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def compute_psnr_ssim(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    pred_01 = pred.detach().clamp(0, 1)
    gt_01 = gt.detach().clamp(0, 1)

    pred_np = tensor_to_uint8_hwc(pred_01)
    gt_np = tensor_to_uint8_hwc(gt_01)
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)

    try:
        ssim_val = ssim(pred_01, gt_01, data_range=1.0, size_average=True).item()
    except TypeError:
        ssim_val = ssim(pred_01, gt_01, size_average=True).item()

    return float(psnr), float(ssim_val)


class MetricPack:
    def __init__(self, device, use_lpips=True, use_niqe=True, use_musiq=True):
        self.device = device

        self.lpips_enabled = False
        self.niqe_enabled = False
        self.musiq_enabled = False

        self.lpips_model = None

        if use_lpips:
            if LPIPS_AVAILABLE:
                self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
                self.lpips_enabled = True
            else:
                print("[WARN] lpips is not installed. LPIPS will be skipped.")

        if use_niqe:
            if SKVIDEO_AVAILABLE:
                self.niqe_enabled = True
            else:
                print("[WARN] scikit-video is not installed. NIQE will be skipped.")

        if use_musiq:
            print("[WARN] MUSIQ is left as N/A in this script because no validated non-pyiqa MUSIQ backend is included.")

    @torch.no_grad()
    def compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
        if not self.lpips_enabled:
            return None
        a = pred * 2.0 - 1.0
        b = gt * 2.0 - 1.0
        return float(self.lpips_model(a, b).mean().item())

    def compute_niqe(self, pred: torch.Tensor) -> Optional[float]:
        if not self.niqe_enabled:
            return None
        gray = rgb_to_gray_uint8(pred)
        val = skvideo_niqe(gray)
        if isinstance(val, np.ndarray):
            val = float(np.mean(val))
        return float(val)

    def compute_musiq(self, pred: torch.Tensor) -> Optional[float]:
        return None


def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return None
    return float(sum(vals) / len(vals))


def save_csv(rows: List[dict], path: str):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_model(model, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if len(missing) > 0:
        print("[WARN] Missing keys:")
        for x in missing:
            print("   ", x)

    if len(unexpected) > 0:
        print("[WARN] Unexpected keys:")
        for x in unexpected:
            print("   ", x)

    print(f"[INFO] Loaded checkpoint from: {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path containing hazy/ and gt/")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--mode", choices=["full", "fullpad", "tile"], default="fullpad")
    ap.add_argument("--pad_mode", choices=["reflect", "replicate", "constant"], default="reflect")
    ap.add_argument("--mod", type=int, default=32)

    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=96)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_ext", choices=["png", "jpg", "jpeg"], default="png")

    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--no_niqe", action="store_true")
    ap.add_argument("--no_musiq", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "dehazed")
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = fusion_net_histoformer2().to(device)
    load_model(model, args.ckpt)
    model.eval()

    ds = PairedTestDataset(args.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    metric_pack = MetricPack(
        device=device,
        use_lpips=not args.no_lpips,
        use_niqe=not args.no_niqe,
        use_musiq=not args.no_musiq
    )

    rows = []

    psnr_list = []
    ssim_list = []
    lpips_list = []
    niqe_list = []
    musiq_list = []

    for hazy, gt, key, hazy_name, gt_name in loader:
        hazy = hazy.to(device)
        gt = gt.to(device)

        if args.mode == "full":
            pred = model(hazy)
        elif args.mode == "fullpad":
            pred = forward_full_pad(model, hazy, mod=args.mod, pad_mode=args.pad_mode)
        else:
            pred = forward_tile_simple(model, hazy, tile=args.tile, overlap=args.overlap, pad_mode=args.pad_mode)

        raw_min = float(pred.min().item())
        raw_max = float(pred.max().item())
        raw_mean = float(pred.mean().item())

        pred = pred.clamp(0.0, 1.0)
        saved_mean = float(pred.mean().item())

        save_name = f"{str(key[0]).zfill(3)}.{args.save_ext}"
        save_path = os.path.join(pred_dir, save_name)
        save_image(pred[0], save_path)

        psnr_val, ssim_val = compute_psnr_ssim(pred, gt)
        lpips_val = metric_pack.compute_lpips(pred, gt)
        niqe_val = metric_pack.compute_niqe(pred)
        musiq_val = metric_pack.compute_musiq(pred)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        niqe_list.append(niqe_val)
        musiq_list.append(musiq_val)

        rows.append({
            "key": key[0],
            "hazy_file": hazy_name[0],
            "gt_file": gt_name[0],
            "saved_file": save_name,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": "" if lpips_val is None else lpips_val,
            "niqe": "" if niqe_val is None else niqe_val,
            "musiq": "" if musiq_val is None else musiq_val,
        })

        if args.sanity:
            niqe_str = "N/A" if niqe_val is None else f"{niqe_val:.4f}"
            musiq_str = "N/A" if musiq_val is None else f"{musiq_val:.4f}"
            print(
                f"key={key[0]:<6} | "
                f"hazy={hazy_name[0]:<18} | "
                f"gt={gt_name[0]:<18} | "
                f"raw[{raw_min:.4f},{raw_max:.4f}] mean={raw_mean:.4f} | "
                f"saved_mean={saved_mean:.4f} | "
                f"PSNR={psnr_val:.4f} SSIM={ssim_val:.6f} | "
                f"NIQE={niqe_str} | "
                f"MUSIQ={musiq_str}"
            )

    avg_psnr = mean_or_none(psnr_list)
    avg_ssim = mean_or_none(ssim_list)
    avg_lpips = mean_or_none(lpips_list)
    avg_niqe = mean_or_none(niqe_list)
    avg_musiq = mean_or_none(musiq_list)

    summary = {
        "num_images": len(ds),
        "checkpoint": args.ckpt,
        "data_root": args.data_root,
        "output_dir": pred_dir,
        "average_psnr": avg_psnr,
        "average_ssim": avg_ssim,
        "average_lpips": avg_lpips,
        "average_niqe": avg_niqe,
        "average_musiq": avg_musiq,
        "note": "NIQE is computed via scikit-video. MUSIQ is left as N/A in this script."
    }

    csv_path = os.path.join(args.out_dir, "metrics_per_image.csv")
    txt_path = os.path.join(args.out_dir, "metrics_summary.txt")
    json_path = os.path.join(args.out_dir, "metrics_summary.json")

    save_csv(rows, csv_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Number of images: {len(ds)}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"Saved dehazed images: {pred_dir}\n\n")
        f.write(f"Average PSNR : {avg_psnr:.6f}\n" if avg_psnr is not None else "Average PSNR : N/A\n")
        f.write(f"Average SSIM : {avg_ssim:.6f}\n" if avg_ssim is not None else "Average SSIM : N/A\n")
        f.write(f"Average LPIPS: {avg_lpips:.6f}\n" if avg_lpips is not None else "Average LPIPS: N/A\n")
        f.write(f"Average NIQE : {avg_niqe:.6f}\n" if avg_niqe is not None else "Average NIQE : N/A\n")
        f.write(f"Average MUSIQ: {avg_musiq:.6f}\n" if avg_musiq is not None else "Average MUSIQ: N/A\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ FINAL AVERAGE OVER TEST SET ================")
    print(f"Number of images : {len(ds)}")
    print(f"Average PSNR     : {avg_psnr:.6f}" if avg_psnr is not None else "Average PSNR     : N/A")
    print(f"Average SSIM     : {avg_ssim:.6f}" if avg_ssim is not None else "Average SSIM     : N/A")
    print(f"Average LPIPS    : {avg_lpips:.6f}" if avg_lpips is not None else "Average LPIPS    : N/A")
    print(f"Average NIQE     : {avg_niqe:.6f}" if avg_niqe is not None else "Average NIQE     : N/A")
    print(f"Average MUSIQ    : {avg_musiq:.6f}" if avg_musiq is not None else "Average MUSIQ    : N/A")
    print("=============================================================\n")

    print(f"[INFO] Dehazed images saved to: {pred_dir}")
    print(f"[INFO] Per-image metrics saved to: {csv_path}")
    print(f"[INFO] Average summary saved to: {txt_path}")
    print(f"[INFO] Average summary JSON saved to: {json_path}")


if __name__ == "__main__":
    main()