import os
import glob
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from model_flashinternimage_histoforme2 import fusion_net_histoformer2
from AdaIR_mix_arch import AdaIR
def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)


class HazyFolderDataset(Dataset):
    def __init__(self, data_root):
        self.tf = transforms.ToTensor()
        hazy_dir = os.path.join(data_root, "hazy")
        if os.path.isdir(hazy_dir):
            self.img_paths = sorted(glob.glob(os.path.join(hazy_dir, "*.*")))
        else:
            self.img_paths = sorted(glob.glob(os.path.join(data_root, "*.*")))

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found under: {data_root} (or {data_root}/hazy)")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)  # [C,H,W] in [0,1]
        return x, os.path.basename(p)


def safe_pad_2d(x, pad_left, pad_right, pad_top, pad_bottom, mode="reflect"):
    """
    reflect has constraint pad <= input_size-1. If invalid, fallback to replicate.
    """
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
def forward_full_pad(model1, model2, x, mod=32, pad_mode="reflect"):
    """
    Pad whole image to a multiple of `mod`, run once, crop back.
    This usually eliminates faint DWT/FFC block artifacts.
    """
    b, c, H, W = x.shape
    assert b == 1

    H2 = ((H + mod - 1) // mod) * mod
    W2 = ((W + mod - 1) // mod) * mod

    pad_bottom = H2 - H
    pad_right  = W2 - W

    xpad = safe_pad_2d(x, 0, pad_right, 0, pad_bottom, mode=pad_mode)
    pred = self_ensemble(xpad, model1)
    pred = self_ensemble(pred, model2)
    pred = pred[:, :, :H, :W]
    return pred


@torch.no_grad()
def forward_tile_simple(model, x, tile=384, overlap=96, pad_mode="reflect"):
    """
    Fallback tile (only if you ever need it). Uses weighted blending + edge padding.
    """
    b, c, H, W = x.shape
    assert b == 1
    stride = tile - overlap

    # precompute 2D weight mask (feather)
    yy = torch.linspace(0, 1, tile, device=x.device)
    xx = torch.linspace(0, 1, tile, device=x.device)
    wy = (1.0 - (2.0 * (yy - 0.5)).abs()).clamp_min(0.0)
    wx = (1.0 - (2.0 * (xx - 0.5)).abs()).clamp_min(0.0)
    w2 = (wy[:, None] * wx[None, :])[None, None, :, :]  # [1,1,tile,tile]

    out = torch.zeros((1, c, H, W), device=x.device, dtype=x.dtype)
    acc = torch.zeros((1, 1, H, W), device=x.device, dtype=x.dtype)

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + tile, H)
            right  = min(left + tile, W)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--mode", choices=["full", "fullpad", "tile"], default="fullpad")
    ap.add_argument("--pad_mode", choices=["reflect", "replicate", "constant"], default="reflect")
    ap.add_argument("--mod", type=int, default=32, help="fullpad: pad H/W to multiple of this")

    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=96)

    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    
    sd = torch.load(args.ckpt, map_location="cpu")

    model1 = fusion_net_histoformer2().to(device)
    model1.load_state_dict(sd["model1"], strict=True)
    model1.eval()

    model2 = AdaIR().to(device)
    model2.load_state_dict(sd["model2"], strict=True)
    model2.eval()

    ds = HazyFolderDataset(args.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for x, name in loader:
        x = x.to(device)

        pred = forward_full_pad(model1, model2, x, mod=args.mod, pad_mode=args.pad_mode)

        raw_min = float(pred.min().item())
        raw_max = float(pred.max().item())
        raw_mean = float(pred.mean().item())

        # IMPORTANT: training target is [0,1] => do NOT tanh-map. Just clamp.
        pred = pred.clamp(0.0, 1.0)
        saved_mean = float(pred.mean().item())

        save_path = os.path.join(args.out_dir, name[0])
        save_image(pred[0], save_path)

        if args.sanity:
            print(f"{name[0]:<14} | raw[{raw_min:.4f},{raw_max:.4f}] mean={raw_mean:.4f} | saved_mean={saved_mean:.4f}")

    print(f"Done. Saved {len(ds)} images to: {args.out_dir}")


if __name__ == "__main__":
    main()
