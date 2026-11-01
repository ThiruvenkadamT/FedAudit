"""Image I/O, head masks, burned-in text detection and the transformations FedAudit uses.

Everything here runs locally at one site on that site's own images.
"""
from __future__ import annotations

import io
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_gray(path: str, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_rgb(path: str, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------- masks
def head_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the head (largest bright connected component, holes filled, convex hull).
    Burned-in text sits in the dark background, so it falls outside this mask."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return np.ones_like(gray, dtype=bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts))
    m = np.zeros_like(comp)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def head_bbox(mask: np.ndarray, pad: int = 4):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    h, w = mask.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1))


def head_crop(gray_or_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    g = gray_or_rgb if gray_or_rgb.ndim == 2 else cv2.cvtColor(gray_or_rgb, cv2.COLOR_RGB2GRAY)
    m = head_mask(g) if mask is None else mask
    x0, y0, x1, y1 = head_bbox(m)
    return gray_or_rgb[y0:y1, x0:x1]


def text_mask(gray: np.ndarray, hmask: np.ndarray | None = None, min_area: int = 12) -> np.ndarray:
    """Burned-in text outside the head: high-contrast thin strokes on dark background.
    Black-hat/top-hat gradient + morphological closing joins characters into word boxes."""
    if hmask is None:
        hmask = head_mask(gray)
    h, w = gray.shape
    k = max(3, (min(h, w) // 64) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k * 3, k * 3)))
    _, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw[cv2.dilate(hmask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0] = 0
    if bw.mean() > 0.25 * 255:  # threshold degenerated on a near-empty background
        return np.zeros_like(gray, dtype=bool)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k * 2 + 1, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(gray, dtype=bool)
    for i in range(1, n):
        x, y, ww, hh, a = stats[i]
        if a < min_area or hh > h * 0.25 or ww > w * 0.9:
            continue
        out[y:y + hh, x:x + ww] = True
    # grow boxes so anti-aliased stroke edges are covered too (a faint rim would still leak the text)
    out = cv2.dilate(out.astype(np.uint8), np.ones((k + 2, k + 2), np.uint8)) > 0
    return out & ~hmask


def has_text(gray: np.ndarray, hmask=None, frac: float = 0.002) -> bool:
    return text_mask(gray, hmask).mean() > frac


# --------------------------------------------------------------------------- near-copy transforms (E3, controlled leakage)
def nc_resize(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    s = rng.uniform(0.5, 0.8)
    small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def nc_jpeg(img: np.ndarray, rng: random.Random) -> np.ndarray:
    q = rng.randint(30, 70)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert(Image.fromarray(img).mode))


def nc_crop(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    f = rng.uniform(0.03, 0.08)
    dx, dy = int(w * f), int(h * f)
    x0, y0 = rng.randint(0, dx), rng.randint(0, dy)
    c = img[y0:h - (dy - y0), x0:w - (dx - x0)]
    return cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)


def nc_overlay(img: np.ndarray, rng: random.Random) -> np.ndarray:
    return paste_random_text(img, rng, n=2)


NEAR_COPY = {"exact": lambda im, r: im.copy(), "resize": nc_resize, "jpeg": nc_jpeg, "crop": nc_crop, "overlay": nc_overlay}


# --------------------------------------------------------------------------- text manipulation (Stage C stats, Stage D augmentation)
def _font(size: int):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def paste_text(img: np.ndarray, text: str, corner: str, size_frac: float = 0.06, value: int = 255) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    h, w = img.shape[:2]
    f = _font(max(8, int(h * size_frac)))
    tw, th = d.textbbox((0, 0), text, font=f)[2:]
    m = int(0.03 * w)
    x = m if "l" in corner else w - tw - m
    y = m if "t" in corner else h - th - m
    fill = value if img.ndim == 2 else (value,) * 3
    d.text((x, y), text, fill=fill, font=f)
    return np.asarray(pil)


def paste_random_text(img: np.ndarray, rng: random.Random, n: int | None = None) -> np.ndarray:
    n = n or rng.randint(1, 3)
    out = img
    for _ in range(n):
        s = "".join(rng.choice(string.ascii_uppercase + string.digits + " :/.") for _ in range(rng.randint(4, 14)))
        out = paste_text(out, s, rng.choice(["tl", "tr", "bl", "br"]), size_frac=rng.uniform(0.04, 0.08),
                         value=rng.randint(180, 255))
    return out


def blank_outside_head(img: np.ndarray, hmask: np.ndarray | None = None) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = head_mask(g) if hmask is None else hmask
    out = img.copy()
    out[~m] = 0
    return out


def blank_text(img: np.ndarray, tmask: np.ndarray | None = None) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t = text_mask(g) if tmask is None else tmask
    out = img.copy()
    out[t] = 0
    return out


PLANTED_MARKER = {0: "R  SCAN-A", 1: "L  SCAN-B"}  # class-linked shortcut for E5 validation


def plant_marker(img: np.ndarray, label: int, n_classes: int) -> np.ndarray:
    txt = PLANTED_MARKER.get(label, f"SERIES {label}") if n_classes == 2 else f"SER-{label:02d}"
    return paste_text(img, txt, "tr", size_frac=0.07)
