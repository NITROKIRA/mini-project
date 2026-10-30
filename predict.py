"""
Banana leaf disease prediction.

    python predict.py banana_leaf.jpg

Steps: OpenCV preprocessing -> ResNet-50 classification -> OpenCV segmentation
(diseased area / severity) -> Grad-CAM. Saves three images next to the input:
    <name>_preprocessed.png   the cleaned image fed to the model
    <name>_severity.png       segmentation overlay (red = diseased, green = healthy)
    <name>_gradcam.png        heatmap of where the model looked (red = most important)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification

from preprocess import preprocess_pil  # OpenCV preprocessing
from severity import compute_severity, make_overlay  # OpenCV segmentation
from gradcam import grad_cam, overlay_heatmap  # where the model looked

# =====================================
# SETTINGS
# =====================================

# Model folder is looked up next to this script, wherever you run it from
MODEL_PATH = str(Path(__file__).resolve().parent / "banana_disease_model")

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
stem = Path(IMAGE_PATH).stem

# =====================================
# LOAD TRAINED MODEL
# =====================================

print("Loading trained model...")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

model = ResNetForImageClassification.from_pretrained(MODEL_PATH)

model.eval()

# =====================================
# LOAD + PREPROCESS IMAGE
# =====================================

image = Image.open(IMAGE_PATH).convert("RGB")

# Same OpenCV steps that were applied to the training images
image = preprocess_pil(image)

image.save(f"{stem}_preprocessed.png")

# Same tensor conversion as training (preprocess_pil already gives 224x224)
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std,
    ),
])

pixel_values = to_tensor(image).unsqueeze(0)

# =====================================
# MAKE PREDICTION
# =====================================

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

confidence, predicted_class = torch.max(probabilities, dim=1)

predicted_label = model.config.id2label[predicted_class.item()]

# =====================================
# SEGMENTATION + SEVERITY
# =====================================

bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

seg = compute_severity(bgr)

cv2.imwrite(f"{stem}_severity.png", make_overlay(bgr, seg["masks"]))

# =====================================
# GRAD-CAM (which areas the model used for its decision)
# =====================================

cam, _ = grad_cam(model, pixel_values, predicted_class.item())

cv2.imwrite(f"{stem}_gradcam.png", overlay_heatmap(bgr, cam))

# =====================================
# DISPLAY RESULT
# =====================================

print("\n=====================================")
print("BANANA LEAF DISEASE PREDICTION")
print("=====================================")
print("Prediction:", predicted_label)
print(f"Confidence: {confidence.item() * 100:.2f}%")
print("-------------------------------------")
if predicted_label.lower() == "healthy" and seg["percent"] > 5:
    print("Note: model says healthy but segmentation found discoloured areas.")
print(f"Diseased area: {seg['percent']:.2f}% of the leaf")
print("Severity:", seg["grade"])
if not seg["reliable"]:
    print("Warning: very little leaf detected - use a closer, clearer photo.")
print("-------------------------------------")
print(f"Saved: {stem}_preprocessed.png, {stem}_severity.png, {stem}_gradcam.png")
print("=====================================")
