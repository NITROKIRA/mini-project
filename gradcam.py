"""
Grad-CAM for the Hugging Face ResNet-50 classifier (no extra package needed).

Shows WHERE in the image the model looked when it chose its prediction.

How it works
  1. Run the image through the model and keep the feature maps of the last
     ResNet stage (the last convolutional block).
  2. Back-propagate the score of the predicted class to get gradients for
     those feature maps.
  3. Average each map's gradient -> an importance weight per feature map.
  4. Weighted sum of the feature maps, keep only positive values (ReLU),
     scale to the image size -> heatmap.

Usage (from predict.py):
    cam, class_id = grad_cam(model, pixel_values)            # predicted class
    overlay = overlay_heatmap(bgr_image, cam)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model, pixel_values: torch.Tensor, class_id: int | None = None):
    """
    model        - ResNetForImageClassification (Hugging Face)
    pixel_values - tensor of shape (1, 3, H, W), already normalised
    class_id     - class to explain; None = the model's own top prediction

    Returns (cam, class_id) where cam is a float32 array in [0, 1] of shape (H, W).
    """
    # last convolutional stage of ResNet-50 (its output is the 7x7x2048 feature map)
    target_layer = model.resnet.encoder.stages[-1]

    store = {}

    def forward_hook(_module, _inputs, output):
        # keep the feature maps (still attached to the graph so we can differentiate)
        store["act"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()

        # gradients are required here, so this must not run under torch.no_grad()
        with torch.enable_grad():
            x = pixel_values.clone().requires_grad_(True)
            logits = model(pixel_values=x).logits

            if class_id is None:
                class_id = int(logits.argmax(dim=1).item())

            # gradient of the class score with respect to the feature maps
            grad = torch.autograd.grad(logits[0, class_id], store["act"])[0]
    finally:
        handle.remove()

    act = store["act"].detach()    # (1, C, h, w)
    grad = grad.detach()           # (1, C, h, w)

    weights = grad.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, h, w)

    cam = F.interpolate(
        cam, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.cpu().numpy().astype(np.float32), class_id


def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """Blend the heatmap (red = most important) on top of the BGR image."""
    h, w = img_bgr.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
