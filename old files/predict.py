import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

# =====================================

# SETTINGS

# =====================================

MODEL_PATH = "banana_disease_model"
IMAGE_PATH = "test.jpg"

# =====================================

# LOAD TRAINED MODEL

# =====================================

print("Loading trained model...")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

model = ResNetForImageClassification.from_pretrained(
MODEL_PATH
)

model.eval()

# =====================================

# LOAD IMAGE

# =====================================

image = Image.open(
IMAGE_PATH
).convert("RGB")

# =====================================

# PROCESS IMAGE

# =====================================

inputs = processor(
images=image,
return_tensors="pt"
)

# =====================================

# MAKE PREDICTION

# =====================================

with torch.no_grad():


	outputs = model(
    		**inputs
	)

probabilities = torch.nn.functional.softmax(
    outputs.logits,
    dim=1
)

confidence, predicted_class = torch.max(
    probabilities,
    dim=1
)


# =====================================

# DISPLAY RESULT

# =====================================

predicted_label = model.config.id2label[
predicted_class.item()
]

print("\n=====================================")
print("BANANA LEAF DISEASE PREDICTION")
print("=====================================")

print("Prediction:", predicted_label)

print(
f"Confidence: {confidence.item() * 100:.2f}%"
)

print("=====================================")
