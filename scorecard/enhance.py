"""
enhance.py — Image enhancement for scorecard row crops.

Modes:
  clahe  — CLAHE adaptive contrast + unsharp-mask sharpening.
            Uses only OpenCV (already installed). Runs in <10 ms per image.
  esrgan — Real-ESRGAN 4x super-resolution, then downscale back to original
            size so the AI gets more detail without increasing API cost.
            Requires torch (CPU-only is fine). ~3–8 s per image on CPU.
  both   — ESRGAN first, then CLAHE on the result.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Download helper ────────────────────────────────────────────────────────────

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "RealESRGAN_x4plus.pth"


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download RealESRGAN_x4plus.pth if not already present."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Real-ESRGAN model to {model_path}")
    total = [0]

    def _progress(count, block, total_size):
        downloaded = count * block
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1_048_576
            print(f"\r  {mb:.1f} MB  ({pct}%)   ", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=_progress)
    print(f"\nModel saved to {model_path}")
    return model_path


# ── CLAHE + unsharp mask ───────────────────────────────────────────────────────

def apply_clahe(img: np.ndarray, clip_limit: float = 3.0, tile: int = 8) -> np.ndarray:
    """
    CLAHE adaptive contrast on the L channel, then unsharp mask sharpening.
    img: BGR uint8.  Returns BGR uint8.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    # Unsharp mask: original * 1.8 − blurred * 0.8
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
    return sharpened


# ── Real-ESRGAN (self-contained — no basicsr/realesrgan package needed) ────────

# RRDBNet architecture, inlined from xinntao/Real-ESRGAN (BSD-3-Clause licence)
def _build_rrdbnet_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                     num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(
                *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
            )
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(
                self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet


_cached_model = None


def load_esrgan_model(model_path: Path = DEFAULT_MODEL_PATH):
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    import torch
    RRDBNet = _build_rrdbnet_class()
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4,
                  num_feat=64, num_block=23, num_grow_ch=32)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    # weights may be stored under 'params_ema', 'params', or at top level
    weights = state.get("params_ema") or state.get("params") or state
    net.load_state_dict(weights, strict=True)
    net.eval()
    _cached_model = net
    print(f"Real-ESRGAN model loaded from {model_path.name}")
    return net


def apply_esrgan(img: np.ndarray, model, tile_size: int = 512,
                 tile_pad: int = 32) -> np.ndarray:
    """
    Run Real-ESRGAN 4x inference, then Lanczos-downscale back to the
    original dimensions.  API call gets the same pixel count but much
    higher effective sharpness/detail.
    img: BGR uint8.  Returns BGR uint8, same size as input.
    """
    import torch
    import torch.nn.functional as F
    import math

    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0
    # HWC BGR → CHW RGB
    tensor = torch.from_numpy(img_f[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0)

    _, c, ih, iw = tensor.shape
    with torch.no_grad():
        if ih <= tile_size and iw <= tile_size:
            out = model(tensor)
        else:
            # Tiled inference for large full-page images
            ow, oh = iw * 4, ih * 4
            out = torch.zeros(1, c, oh, ow)
            tx = math.ceil(iw / tile_size)
            ty = math.ceil(ih / tile_size)
            for iy in range(ty):
                for ix in range(tx):
                    y0 = max(0, iy * tile_size - tile_pad)
                    y1 = min(ih, (iy + 1) * tile_size + tile_pad)
                    x0 = max(0, ix * tile_size - tile_pad)
                    x1 = min(iw, (ix + 1) * tile_size + tile_pad)
                    tile_out = model(tensor[:, :, y0:y1, x0:x1])
                    # Crop padding from output
                    py0 = (iy * tile_size - y0) * 4
                    py1 = py0 + min(tile_size, ih - iy * tile_size) * 4
                    px0 = (ix * tile_size - x0) * 4
                    px1 = px0 + min(tile_size, iw - ix * tile_size) * 4
                    dy0, dy1 = iy * tile_size * 4, min(oh, (iy+1) * tile_size * 4)
                    dx0, dx1 = ix * tile_size * 4, min(ow, (ix+1) * tile_size * 4)
                    out[:, :, dy0:dy1, dx0:dx1] = tile_out[:, :, py0:py1, px0:px1]

    # CHW RGB → HWC BGR, clip, uint8
    out_np = out.squeeze(0).permute(1, 2, 0).numpy()
    out_np = np.clip(out_np, 0, 1)
    out_bgr = (out_np[:, :, ::-1] * 255.0).round().astype(np.uint8)
    # Downscale back to original dimensions
    return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ── Hough diagonal suppression ────────────────────────────────────────────────

def suppress_diagonals(
    img: np.ndarray,
    min_length: int = 80,
    max_gap: int = 6,
    angle_lo: float = 28.0,
    angle_hi: float = 62.0,
    erase_width: int = 5,
    debug: bool = False,
) -> np.ndarray:
    """
    Detect long diagonal lines with the Probabilistic Hough transform and
    paint them white.

    Why this works:
      - Template diagonals span an entire PA cell corner-to-corner.
        At ~95 px wide x ~100 px tall per cell, that is ~138 px long.
      - X marks for 2B/3B are written inside a single sub-cell (~47x50 px),
        so each stroke is only ~69 px long.
      - Setting min_length=80 catches the template lines but leaves X marks
        (and all short handwritten strokes) untouched.
      - angle_lo/hi restrict detection to the 28-62 degree diagonal band,
        leaving vertical (hit strokes) and horizontal marks alone.

    Parameters
    ----------
    min_length   : minimum pixel length for a line to be suppressed (tune up
                   if real marks are erased; tune down if diagonals survive)
    max_gap      : max gap in pixels still treated as one line
    angle_lo/hi  : angular range (degrees) for diagonal lines to suppress
    erase_width  : thickness of the white erasure stroke (px)
    debug        : if True, draw suppressed lines in bright red so you can
                   inspect which lines were found before committing
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny with moderate thresholds; blue pen on white paper has good contrast
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )

    result = img.copy()
    if lines is None:
        return result

    erase_color = (0, 0, 255) if debug else (255, 255, 255)
    n_erased = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1:
            continue  # vertical — skip
        angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
        if angle_lo <= angle <= angle_hi:
            cv2.line(result, (x1, y1), (x2, y2), erase_color, erase_width)
            n_erased += 1

    if debug:
        print(f"suppress_diagonals: erased {n_erased} line segments (shown in red)")
    return result


# ── Public entry point ─────────────────────────────────────────────────────────

def enhance_image(img: np.ndarray, mode: str,
                  model_path: Path = DEFAULT_MODEL_PATH) -> np.ndarray:
    """
    mode:
      'clahe'       — CLAHE + unsharp mask (fast, OpenCV only)
      'esrgan'      — Real-ESRGAN 4x then downscale (needs torch)
      'both'        — ESRGAN then CLAHE
      'nodiag'      — Hough diagonal suppression (removes long template diagonals)
      'nodiag+esrgan' — diagonal suppression then ESRGAN sharpening
    Returns enhanced BGR uint8 image, same size as input.
    """
    if mode == "clahe":
        return apply_clahe(img)
    if mode == "esrgan":
        model = load_esrgan_model(model_path)
        return apply_esrgan(img, model)
    if mode == "both":
        model = load_esrgan_model(model_path)
        return apply_clahe(apply_esrgan(img, model))
    if mode == "nodiag":
        return suppress_diagonals(img)
    if mode == "nodiag+esrgan":
        model = load_esrgan_model(model_path)
        return apply_esrgan(suppress_diagonals(img), model)
    raise ValueError(f"Unknown enhance mode: {mode!r}")


# ── CLI: python enhance.py <image> [clahe|esrgan|both] ────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enhance.py <image_path> [clahe|esrgan|both]")
        sys.exit(1)
    src = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    img = cv2.imread(str(src))
    if img is None:
        print(f"Cannot read {src}")
        sys.exit(1)
    if mode in ("esrgan", "both"):
        ensure_model()
    out = enhance_image(img, mode)
    dst = src.parent / f"{src.stem}_{mode}{src.suffix}"
    cv2.imwrite(str(dst), out)
    print(f"Saved → {dst}")
