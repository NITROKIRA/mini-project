import torch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
from tqdm import tqdm

# =====================================

# 1. SETTINGS

# =====================================

DATASET_PATH = "dataset"
MODEL_NAME = "microsoft/resnet-50"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001

# Reproducible dataset split

torch.manual_seed(42)

# =====================================

# 2. SELECT DEVICE

# =====================================

device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

if torch.cuda.is_available():
print("GPU:", torch.cuda.get_device_name(0))

# =====================================

# 3. LOAD IMAGE PROCESSOR

# =====================================

processor = AutoImageProcessor.from_pretrained(
MODEL_NAME
)

# =====================================

# 4. IMAGE TRANSFORMATIONS

# =====================================

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=processor.image_mean,
std=processor.image_std
)
])

# =====================================

# 5. LOAD DATASET

# =====================================

dataset = datasets.ImageFolder(
DATASET_PATH,
transform=transform
)

print("\nClasses:")
print(dataset.classes)

print("\nNumber of images:")
print(len(dataset))

# =====================================

# 6. SPLIT DATASET

# =====================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
dataset,
[train_size, val_size]
)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# =====================================

# 7. CREATE DATA LOADERS

# =====================================

train_loader = DataLoader(
train_dataset,
batch_size=BATCH_SIZE,
shuffle=True
)

val_loader = DataLoader(
val_dataset,
batch_size=BATCH_SIZE,
shuffle=False
)

# =====================================

# 8. CREATE LABEL MAPS

# =====================================

num_labels = len(dataset.classes)

id2label = {
i: label
for i, label in enumerate(dataset.classes)
}

label2id = {
label: i
for i, label in enumerate(dataset.classes)
}

# =====================================

# 9. LOAD PRETRAINED RESNET-50

# =====================================

print("\nLoading ResNet-50 model...")

model = ResNetForImageClassification.from_pretrained(
MODEL_NAME,
num_labels=num_labels,
id2label=id2label,
label2id=label2id,
ignore_mismatched_sizes=True
)

model.to(device)

# =====================================

# 10. OPTIMIZER

# =====================================

optimizer = torch.optim.AdamW(
model.parameters(),
lr=LEARNING_RATE
)

# =====================================

# 11. TRAINING

# =====================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

```
model.train()

running_loss = 0.0

progress_bar = tqdm(
    train_loader,
    desc=f"Epoch {epoch + 1}/{EPOCHS}"
)

for images, labels in progress_bar:

    # Move data to CPU/GPU
    images = images.to(device)
    labels = labels.to(device)

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(
        pixel_values=images,
        labels=labels
    )

    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Update model weights
    optimizer.step()

    running_loss += loss.item()

    # Show current loss
    progress_bar.set_postfix(
        loss=f"{loss.item():.4f}"
    )

average_loss = running_loss / len(train_loader)

print(
    f"\nEpoch {epoch + 1} completed"
)

print(
    f"Average Training Loss: {average_loss:.4f}"
)
```

# =====================================

# 12. VALIDATION

# =====================================

print("\nStarting validation...")

model.eval()

correct = 0
total = 0

with torch.no_grad():

```
progress_bar = tqdm(
    val_loader,
    desc="Validating"
)

for images, labels in progress_bar:

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(
        pixel_values=images
    )

    predictions = torch.argmax(
        outputs.logits,
        dim=1
    )

    correct += (
        predictions == labels
    ).sum().item()

    total += labels.size(0)
```

accuracy = 100 * correct / total

print("\n=====================================")
print("VALIDATION RESULTS")
print("=====================================")
print(f"Correct Predictions: {correct}")
print(f"Total Images: {total}")
print(f"Validation Accuracy: {accuracy:.2f}%")
print("=====================================")

# =====================================

# 13. SAVE TRAINED MODEL

# =====================================

SAVE_FOLDER = "banana_disease_model"

print("\nSaving trained model...")

model.save_pretrained(
SAVE_FOLDER
)

processor.save_pretrained(
SAVE_FOLDER
)

print("\n=====================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=====================================")
print(f"Model saved in: {SAVE_FOLDER}")
