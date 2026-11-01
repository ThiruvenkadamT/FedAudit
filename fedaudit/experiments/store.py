"""Decoded-image caches so 400+ training jobs do not re-decode JPEGs.

store = {'ids': array, 'img': uint8 (N,H,W[,3]), optional 'hmask','tmask' (bool, N,H,W)}
saved as .npy files in one folder, memory-mapped by workers.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..data import imaging as im


def _read(path, size, rgb):
    img = Image.open(path).convert("RGB" if rgb else "L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, np.uint8)


def build_store(df: pd.DataFrame, folder: str, size: int, rgb: bool = False, masks: bool = False, workers: int = 4):
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(os.path.join(folder, "ids.npy")):
        ids = np.load(os.path.join(folder, "ids.npy"), allow_pickle=True)
        if set(df["image_id"]) <= set(ids):
            return folder
    paths = df["path"].tolist()
    with ThreadPoolExecutor(workers) as ex:
        imgs = list(ex.map(lambda p: _read(p, size, rgb), paths))
    arr = np.stack(imgs)
    np.save(os.path.join(folder, "img.npy"), arr)
    np.save(os.path.join(folder, "ids.npy"), df["image_id"].values.astype(object), allow_pickle=True)
    df[["image_id", "path", "label", "label_name", "site", "split", "group", "modality"]].to_csv(os.path.join(folder, "rows.csv"), index=False)
    if masks:
        g = arr if arr.ndim == 3 else arr.mean(-1).astype(np.uint8)

        def mk(x):
            h = im.head_mask(x)
            return h, im.text_mask(x, h)

        with ThreadPoolExecutor(workers) as ex:
            hm = list(ex.map(mk, list(g)))
        np.save(os.path.join(folder, "hmask.npy"), np.packbits(np.stack([a for a, _ in hm]), axis=-1))
        np.save(os.path.join(folder, "tmask.npy"), np.packbits(np.stack([b for _, b in hm]), axis=-1))
    return folder


class Store:
    def __init__(self, folder: str):
        self.folder = folder
        self.ids = np.load(os.path.join(folder, "ids.npy"), allow_pickle=True)
        self.pos = {k: i for i, k in enumerate(self.ids)}
        self.img = np.load(os.path.join(folder, "img.npy"), mmap_mode="r")
        hm = os.path.join(folder, "hmask.npy")
        self._h = np.load(hm, mmap_mode="r") if os.path.exists(hm) else None
        tm = os.path.join(folder, "tmask.npy")
        self._t = np.load(tm, mmap_mode="r") if os.path.exists(tm) else None
        self.size = self.img.shape[1]

    def get(self, image_id):
        return np.array(self.img[self.pos[image_id]])

    def masks(self, image_id):
        i = self.pos[image_id]
        h = np.unpackbits(self._h[i], axis=-1)[:, :self.size].astype(bool)
        t = np.unpackbits(self._t[i], axis=-1)[:, :self.size].astype(bool)
        return h, t


def fit(a: np.ndarray, size: int) -> np.ndarray:
    return a if a.shape[:2] == (size, size) else cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
