"""Build a small fake /kaggle/input tree that mimics the folder layouts of the six public datasets,
with known cross-source copies (like Nickparvar containing SARTAJ/figshare/Br35H) and burned-in
text on CT images. Used to test the full pipeline without downloading anything.

    python tests/make_fake_kaggle_input.py /path/to/fake_input [n_per_class]
"""
import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedaudit.data.imaging import NEAR_COPY, paste_text  # noqa: E402
from fedaudit.data.synthetic import _head  # noqa: E402


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def main(root, n=24, size=160, seed=0):
    rng = np.random.default_rng(seed)
    pr = random.Random(seed)
    cls4 = ["glioma", "meningioma", "notumor", "pituitary"]
    lab_of = {"glioma": 1, "meningioma": 2, "notumor": 0, "pituitary": 3}
    pool = {}  # source -> list of (img, class)
    gains = {"figshare": 0.9, "sartaj": 1.0, "br35h": 1.1}
    for src, classes in (("figshare", ["glioma", "meningioma", "pituitary"]), ("sartaj", cls4), ("br35h", ["notumor", "tumor"])):
        pool[src] = [(_head(rng, size, lab_of.get(c, 1), 4, gains[src]), c) for c in classes for _ in range(n)]
    for i, (img, c) in enumerate(pool["figshare"]):
        save(img, f"{root}/figshare-brain-tumor/{c}/{i}.jpg")
    for i, (img, c) in enumerate(pool["sartaj"]):
        sp = "Testing" if i % 5 == 0 else "Training"
        save(img, f"{root}/brain-tumor-classification-mri/{sp}/{c}_tumor/{i}.jpg" if c != "notumor" else
             f"{root}/brain-tumor-classification-mri/{sp}/no_tumor/{i}.jpg")
    for i, (img, c) in enumerate(pool["br35h"]):
        save(img, f"{root}/brain-tumor-detection/{'no' if c == 'notumor' else 'yes'}/{i}.jpg")
    # Nickparvar: own images + ~60% copies from the other sources (some re-encoded)
    nick = [(_head(rng, size, lab_of[c], 4, 1.05), c) for c in cls4 for _ in range(n // 2)]
    for src in ("figshare", "sartaj", "br35h"):
        for img, c in pool[src]:
            if c in cls4 and pr.random() < 0.6:
                kind = pr.choice(["exact", "exact", "jpeg", "resize"])
                nick.append((NEAR_COPY[kind](img, pr), c))
    for i, (img, c) in enumerate(nick):
        sp = "Testing" if i % 5 == 0 else "Training"
        save(img, f"{root}/brain-tumor-mri-dataset/{sp}/{c}/{i}.jpg")
    # HAM10000: RGB lesions, 4 sources, several images per lesion
    rows = []
    dx = ["nv", "mel", "bkl", "bcc"]
    k = 0
    for src in ("vidir_modern", "rosendahl", "vidir_molemax", "vienna_dias"):
        for les in range(n):
            d = pr.choice(dx)
            base = np.zeros((size, size, 3), np.uint8) + np.array([200, 160, 140], np.uint8)
            cx, cy, r = pr.randint(50, 110), pr.randint(50, 110), 15 + 5 * dx.index(d)
            yy, xx = np.ogrid[:size, :size]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
            base[m] = np.array([90 - 10 * dx.index(d), 50, 40], np.uint8)
            noise = rng.integers(0, 25, base.shape, dtype=np.uint8)
            for v in range(pr.choice([1, 1, 2, 3])):
                img = np.clip(base.astype(int) + noise - 12 + (v * 7), 0, 255).astype(np.uint8)
                iid = f"ISIC_{k:07d}"
                k += 1
                save(img, f"{root}/skin-cancer-mnist-ham10000/HAM10000_images_part_1/{iid}.jpg")
                rows.append(dict(lesion_id=f"HAM_{src}_{les}", image_id=iid, dx=d, dx_type="histo", age=50, sex="male",
                                 localization="back", dataset=src))
    pd.DataFrame(rows).to_csv(f"{root}/skin-cancer-mnist-ham10000/HAM10000_metadata.csv", index=False)
    # CT/MRI multimodal: CT images carry burned-in text, tumor-linked layout
    for mod, folder in (("CT", "Brain Tumor CT scan Images"), ("MRI", "Brain Tumor MRI images")):
        for c in ("Healthy", "Tumor"):
            for i in range(n * 2):
                img = _head(rng, size, 1 if c == "Tumor" else 0, 2, 1.0 if mod == "MRI" else 0.8)
                if mod == "CT":
                    img = paste_text(img, f"ID {pr.randint(1000, 9999)}  {'KV 120' if c == 'Tumor' else 'KV 100'}", "tl", 0.06)
                    img = paste_text(img, "W:80 L:40", "br", 0.05)
                save(img, f"{root}/brain-tumor-multimodal-image-ct-and-mri/Dataset/{folder}/{c}/{c.lower()} ({i}).jpg")
    print("fake input written to", root)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
