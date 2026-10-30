"""
Colour-based segmentation + disease severity for banana leaves (OpenCV, HSV).

Two segmentation steps:
  1. LEAF mask      - which pixels are leaf (green, yellow or brown tissue)
                      versus background (sky, soil, white glare, grey)
  2. DISEASE mask   - inside the leaf, which pixels are NOT healthy green
                      (yellow halos, brown/black lesions)

    severity (%) = disease pixels / leaf pixels * 100

OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
These thresholds are a starting point. Banana leaves in different lighting may
need small adjustments (see the constants below).
"""

import cv2
import numpy as np

# ---- thresholds you can tune -------------------------------------------------
HEALTHY_H = (30, 90)      # green hue range
HEALTHY_S_MIN = 15        # low value so pale grey-green leaves still count as leaf
HEALTHY_V_MIN = 50

YELLOW_H = (15, 30)       # yellow halos, chlorosis
BROWN_H = (0, 15)         # brown / necrotic tissue
DISEASE_S_MIN = 100       # lesions are strongly coloured; this ignores faint tints
DISEASE_V_MIN = 40

DARK_V_MAX = 55           # very dark pixels inside the leaf = black lesions

MIN_LEAF_FRACTION = 0.05  # if less than this is "leaf", segmentation is unreliable

# severity bands (percent of leaf area) - common convention, adjust to your reference
GRADES = [
    (1.0, "None / trace"),
    (10.0, "Mild"),
    (30.0, "Moderate"),
    (100.1, "Severe"),
]
# ------------------------------------------------------------------------------


def _clean(mask: np.ndarray, open_px: int = 3, close_px: int = 7) -> np.ndarray:
    """Remove speckle noise and fill tiny holes."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def segment_leaf(img_bgr: np.ndarray):
    """Return (leaf_mask, healthy_mask, disease_mask), each uint8 0/255."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    healthy = (
        (h >= HEALTHY_H[0]) & (h <= HEALTHY_H[1])
        & (s >= HEALTHY_S_MIN) & (v >= HEALTHY_V_MIN)
    )
    yellow = (
        (h >= YELLOW_H[0]) & (h < YELLOW_H[1])
        & (s >= DISEASE_S_MIN) & (v >= DISEASE_V_MIN)
    )
    brown = (
        (h >= BROWN_H[0]) & (h < BROWN_H[1])
        & (s >= DISEASE_S_MIN) & (v >= DISEASE_V_MIN)
    )

    healthy = _clean(healthy.astype(np.uint8) * 255)
    diseased = _clean((yellow | brown).astype(np.uint8) * 255)

    # leaf = healthy tissue + diseased tissue, with gaps closed so lesions
    # surrounded by leaf (and dark spots inside it) are counted as leaf
    leaf = cv2.bitwise_or(healthy, diseased)
    leaf = cv2.morphologyEx(
        leaf, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )

    # very dark pixels inside the leaf are black lesions (not background)
    dark = ((v <= DARK_V_MAX) & (leaf > 0)).astype(np.uint8) * 255
    diseased = cv2.bitwise_or(diseased, _clean(dark))

    # disease only counts inside the leaf, and never overlaps healthy tissue
    diseased = cv2.bitwise_and(diseased, leaf)
    healthy = cv2.bitwise_and(healthy, cv2.bitwise_not(diseased))

    return leaf, healthy, diseased


def grade_for(percent: float) -> str:
    for upper, name in GRADES:
        if percent < upper:
            return name
    return GRADES[-1][1]


def compute_severity(img_bgr: np.ndarray) -> dict:
    """
    Returns a dict with:
      percent       - diseased area as % of leaf area
      grade         - None / Mild / Moderate / Severe
      leaf_fraction - share of the image classed as leaf
      reliable      - False if almost no leaf was found (check the photo)
      masks         - (leaf, healthy, disease)
    """
    leaf, healthy, diseased = segment_leaf(img_bgr)

    leaf_px = int(np.count_nonzero(leaf))
    dis_px = int(np.count_nonzero(diseased))
    total = img_bgr.shape[0] * img_bgr.shape[1]

    leaf_fraction = leaf_px / total
    percent = 100.0 * dis_px / leaf_px if leaf_px else 0.0

    return {
        "percent": round(percent, 2),
        "grade": grade_for(percent),
        "leaf_fraction": round(leaf_fraction, 3),
        "reliable": leaf_fraction >= MIN_LEAF_FRACTION,
        "masks": (leaf, healthy, diseased),
    }


def make_overlay(img_bgr: np.ndarray, masks, alpha: float = 0.45) -> np.ndarray:
    """Original image with diseased pixels in red, healthy in green, background dimmed."""
    leaf, healthy, diseased = masks
    out = img_bgr.copy()

    colour = np.zeros_like(out)
    colour[healthy > 0] = (0, 200, 0)      # green
    colour[diseased > 0] = (0, 0, 255)     # red
    tinted = cv2.addWeighted(out, 1 - alpha, colour, alpha, 0)

    out = np.where((leaf > 0)[..., None], tinted, (out * 0.35).astype(np.uint8))

    # outline the diseased regions
    contours, _ = cv2.findContours(diseased, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


if __name__ == "__main__":
    # Standalone check:  python severity.py leaf.jpg
    import sys

    from preprocess import preprocess_bgr

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    original = cv2.imread(path)
    if original is None:
        raise SystemExit(f"Could not read image: {path}")

    img = preprocess_bgr(original)
    result = compute_severity(img)
    cv2.imwrite("severity_overlay.png", make_overlay(img, result["masks"]))

    print(f"Diseased area : {result['percent']}% of the leaf")
    print(f"Severity grade: {result['grade']}")
    if not result["reliable"]:
        print("Warning: very little leaf was detected - use a closer, clearer photo.")
    print("Saved severity_overlay.png")
