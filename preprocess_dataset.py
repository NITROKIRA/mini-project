"""
Run ONCE before training. Reads every image in dataset/<class>/ and writes the
OpenCV-preprocessed version to dataset_preprocessed/<class>/, so training
does not repeat the (slow) denoising step every epoch.

    python preprocess_dataset.py
"""

import os

import cv2

from preprocess import preprocess_bgr

SRC = "dataset"
DST = "dataset_preprocessed"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

done = skipped = 0

for cls in sorted(os.listdir(SRC)):
    src_dir = os.path.join(SRC, cls)
    if not os.path.isdir(src_dir):
        continue
    dst_dir = os.path.join(DST, cls)
    os.makedirs(dst_dir, exist_ok=True)

    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith(EXTS):
            continue
        img = cv2.imread(os.path.join(src_dir, name))
        if img is None:
            print("Skipping unreadable file:", os.path.join(src_dir, name))
            skipped += 1
            continue
        out = preprocess_bgr(img)
        # PNG = lossless, so no extra JPEG artefacts are added after cleaning
        base = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(dst_dir, base + ".png"), out)
        done += 1

    print(f"{cls}: done")

print(f"\nProcessed {done} images, skipped {skipped}. Output: {DST}/")
