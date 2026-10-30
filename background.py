"""
Background removal / leaf cropping with OpenCV GrabCut.

Why GrabCut and not "keep the green pixels"?
  Diseased tissue is yellow or brown, and some leaves are grey-green, so any
  rule based only on green would delete real leaf (and the disease itself).
  GrabCut looks at colour AND structure: it learns what the "main object"
  and the "background" look like in THIS photo.

Safety net
  If the result looks wrong (almost nothing kept, or almost nothing removed),
  the original image is returned unchanged. Close-up leaf photos that already
  fill the frame therefore pass through untouched.

    cleaned = remove_background(img_bgr, mode="crop")
        mode="crop"  - crop to the leaf's bounding box (background at the edges is cut off)
        mode="mask"  - keep image size, paint the background a neutral grey
        mode="both"  - mask the background, then crop (recommended)
"""

import cv2
import numpy as np

WORK_SIZE = 400          # GrabCut runs on a downscaled copy for speed
BORDER_FRAC = 0.04       # outer border of the photo is assumed to be background
MIN_FG_FRACTION = 0.20   # less than this kept -> segmentation not trusted
MAX_FG_FRACTION = 0.92   # more than this kept -> nothing worth removing
CROP_MARGIN = 0.03       # extra space around the leaf when cropping
NEUTRAL = (128, 128, 128)  # colour used to paint the removed background


def _grabcut_mask(small_bgr: np.ndarray) -> np.ndarray:
    """Return a 0/1 foreground mask for the downscaled image."""
    h, w = small_bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)  # default: probably leaf

    # outer border: sure background
    by, bx = int(h * BORDER_FRAC), int(w * BORDER_FRAC)
    mask[:by, :] = cv2.GC_BGD
    mask[-by:, :] = cv2.GC_BGD
    mask[:, :bx] = cv2.GC_BGD
    mask[:, -bx:] = cv2.GC_BGD

    # blue sky is certainly not a banana leaf (OpenCV hue 100-130)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    sky = (hsv[..., 0] >= 100) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 40)
    mask[sky] = cv2.GC_BGD

    # centre of the photo: the leaf is normally the main subject
    cy0, cy1 = int(h * 0.3), int(h * 0.7)
    cx0, cx1 = int(w * 0.3), int(w * 0.7)
    centre = mask[cy0:cy1, cx0:cx1]
    centre[centre == cv2.GC_PR_FGD] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small_bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)

    # tidy: remove specks, fill holes, keep the largest piece(s)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        keep = np.zeros_like(fg)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for i in range(1, n):
            # keep the biggest region and any other region at least 15% of it
            if i == largest or stats[i, cv2.CC_STAT_AREA] >= 0.15 * stats[largest, cv2.CC_STAT_AREA]:
                keep[labels == i] = 1
        fg = keep
    return fg


def leaf_mask(img_bgr: np.ndarray):
    """
    Full-size 0/255 leaf mask, or None if the segmentation should not be trusted.
    """
    h, w = img_bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else img_bgr.copy()

    fg = _grabcut_mask(small)
    frac = float(fg.mean())
    if frac < MIN_FG_FRACTION or frac > MAX_FG_FRACTION:
        return None

    return cv2.resize(fg * 255, (w, h), interpolation=cv2.INTER_NEAREST)


def remove_background(img_bgr: np.ndarray, mode: str = "both") -> np.ndarray:
    """Remove the background around the leaf. Returns the original if unsure."""
    mask = leaf_mask(img_bgr)
    if mask is None:
        return img_bgr

    out = img_bgr
    if mode in ("mask", "both"):
        out = img_bgr.copy()
        out[mask == 0] = NEUTRAL

    if mode in ("crop", "both"):
        ys, xs = np.where(mask > 0)
        h, w = img_bgr.shape[:2]
        my, mx = int(h * CROP_MARGIN), int(w * CROP_MARGIN)
        y0, y1 = max(0, ys.min() - my), min(h, ys.max() + 1 + my)
        x0, x1 = max(0, xs.min() - mx), min(w, xs.max() + 1 + mx)
        out = out[y0:y1, x0:x1]

    return out


if __name__ == "__main__":
    # Quick check:  python background.py leaf.jpg
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    out = remove_background(img, mode="both")
    if out is img:
        print("Background removal skipped (leaf already fills the photo, or result not reliable).")
    cv2.imwrite("background_removed.png", out)
    print("Saved background_removed.png")
