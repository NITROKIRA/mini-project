"""
OpenCV preprocessing for banana leaf disease detection.

The SAME function must be used for training and prediction, otherwise the model
sees differently-processed images at test time than it was trained on.

Pipeline (in order):
  1. Resize to 224x224           -> ResNet-50 input size
  2. Denoise (non-local means)   -> removes camera/JPEG noise, keeps lesion edges
  3. CLAHE on the L channel      -> evens out sun/shadow without shifting colours
  4. Mild unsharp mask           -> makes spot borders clearer

Colour matters a lot here (yellow halos, brown/black spots), so contrast is
adjusted only on the lightness channel of LAB, never on R/G/B separately.
"""

import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 224


def preprocess_bgr(
    img_bgr: np.ndarray,
    size: int = IMG_SIZE,
    denoise: bool = True,
    clahe: bool = True,
    sharpen: bool = True,
) -> np.ndarray:
    """Take a BGR uint8 image (as read by cv2.imread) and return a BGR uint8 image."""

    # 1. Resize (INTER_AREA is best for shrinking, INTER_CUBIC for enlarging)
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    img = cv2.resize(img_bgr, (size, size), interpolation=interp)

    # 2. Denoise - keep h small so lesion texture is not smoothed away
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # 3. CLAHE on lightness only
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 4. Unsharp mask: original + 0.5 * (original - blurred)
    if sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


def preprocess_pil(pil_img: Image.Image, **kwargs) -> Image.Image:
    """PIL RGB in -> PIL RGB out, so it can sit inside a torchvision Compose."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = preprocess_bgr(bgr, **kwargs)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


class OpenCVPreprocess:
    """Drop-in torchvision transform (use it FIRST in transforms.Compose)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        return preprocess_pil(pil_img, **self.kwargs)

    def __repr__(self):
        return f"OpenCVPreprocess({self.kwargs})"


if __name__ == "__main__":
    # Quick visual check:  python preprocess.py some_leaf.jpg
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    original = cv2.imread(path)
    if original is None:
        raise SystemExit(f"Could not read image: {path}")
    processed = preprocess_bgr(original)
    original_224 = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite("preprocess_preview.png", np.hstack([original_224, processed]))
    print("Saved preprocess_preview.png (left = original, right = preprocessed)")
