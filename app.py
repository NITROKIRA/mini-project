"""
Streamlit web app for banana leaf disease detection.

Run with:
    streamlit run app.py

The app reuses the exact pipeline from predict2.py:
OpenCV preprocessing -> ResNet-50 classification -> OpenCV severity -> Grad-CAM.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification

from background import remove_background
from gradcam import grad_cam, overlay_heatmap
from preprocess import preprocess_pil
from severity import compute_severity, make_overlay

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "banana_disease_model")
SAMPLE_IMAGE = BASE_DIR / "banana_leaf.jpg"

CLASS_INFO = {
    "healthy": ("Healthy", "No disease detected. Leaf tissue looks normal."),
    "cordana": ("Cordana Leaf Spot", "Dark brown oval spots caused by Cordana musae."),
    "pestalotiopsis": ("Pestalotiopsis", "Brown/grey lesions with dark margins on the leaf."),
    "sigatoka": ("Sigatoka", "Yellow-streaked lesions that turn brown/black (black Sigatoka)."),
}

st.set_page_config(
    page_title="Banana Leaf Disease Detection",
    page_icon="ðŸŒ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1200px; }
        .result-card {
            padding: 1.1rem 1.3rem;
            border-radius: 12px;
            background: #f0f7ee;
            border-left: 6px solid #2e7d32;
            margin-bottom: 1rem;
        }
        .result-card h2 { margin: 0 0 0.25rem 0; color: #1b5e20; }
        .result-card p { margin: 0; color: #33413a; }
        .small-note { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = ResNetForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model


def to_tensor(image: Image.Image, processor) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
    return transform(image).unsqueeze(0)


def analyse(pil_image: Image.Image, processor, model, remove_bg: bool):
    image = pil_image.convert("RGB")
    removed = None

    if remove_bg:
        bgr_full = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cut = remove_background(bgr_full, mode="both")
        if cut is not bgr_full:
            removed = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(removed)

    processed = preprocess_pil(image)
    pixel_values = to_tensor(processed, processor)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    confidence, predicted_class = torch.max(probabilities, dim=0)
    predicted_label = model.config.id2label[predicted_class.item()]

    bgr = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)
    seg = compute_severity(bgr)
    severity_img = cv2.cvtColor(make_overlay(bgr, seg["masks"]), cv2.COLOR_BGR2RGB)

    cam, _ = grad_cam(model, pixel_values, predicted_class.item())
    gradcam_img = cv2.cvtColor(overlay_heatmap(bgr, cam), cv2.COLOR_BGR2RGB)

    all_scores = {
        model.config.id2label[i]: float(probabilities[i]) for i in range(len(probabilities))
    }

    return {
        "original": np.array(pil_image.convert("RGB")),
        "removed": removed,
        "processed": np.array(processed),
        "severity": severity_img,
        "gradcam": gradcam_img,
        "label": predicted_label,
        "confidence": float(confidence),
        "scores": all_scores,
        "seg": seg,
    }


def encode_png(array_rgb: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR))
    return buffer.tobytes() if ok else b""


st.title("ðŸŒ Banana Leaf Disease Detection")
st.caption(
    "Upload a banana leaf photo. The ResNet-50 classifier names the disease, "
    "OpenCV estimates how much of the leaf is affected, and Grad-CAM shows where the model looked."
)

with st.sidebar:
    st.header("Settings")
    remove_bg = st.checkbox(
        "Remove background",
        value=False,
        help="Use for field photos with sky, soil or other leaves behind the subject.",
    )
    st.divider()
    st.subheader("Model")
    st.markdown(
        "- **Architecture:** ResNet-50\n"
        "- **Classes:** Cordana, Healthy, Pestalotiopsis, Sigatoka\n"
        "- **Input:** 224 Ã— 224 after OpenCV cleaning"
    )
    st.divider()
    st.caption(
        "This is a decision-support tool, not a replacement for expert diagnosis. "
        "Use a clear, close-up photo of a single leaf for best results."
    )

source = st.radio(
    "Choose an image source",
    ["Upload an image", "Use sample image"],
    horizontal=True,
)

uploaded = None
sample_used = False
if source == "Upload an image":
    uploaded = st.file_uploader("Upload a banana leaf image", type=["jpg", "jpeg", "png"])
else:
    if SAMPLE_IMAGE.exists():
        sample_used = True
    else:
        st.warning("Sample image not found in the project folder.")

pil_image = None
if uploaded is not None:
    pil_image = Image.open(uploaded)
elif sample_used:
    pil_image = Image.open(SAMPLE_IMAGE)

if pil_image is None:
    st.info("Upload a leaf photo to start the analysis.")
    st.stop()

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Input")
    st.image(pil_image, width="stretch")

if st.button("Analyse leaf", type="primary", width="stretch"):
    processor, model = load_model()

    with st.spinner("Analysing..."):
        result = analyse(pil_image, processor, model, remove_bg)

    st.session_state["result"] = result

result = st.session_state.get("result")

if result is None:
    st.stop()

label = result["label"]
pretty_name, description = CLASS_INFO.get(label, (label.title(), ""))
confidence = result["confidence"] * 100
seg = result["seg"]

with right:
    st.subheader("Diagnosis")
    st.markdown(
        f"""
        <div class="result-card">
            <h2>{pretty_name}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{confidence:.1f}%")
    c2.metric("Diseased area", f"{seg['percent']:.1f}%")
    c3.metric("Severity", seg["grade"])

    if label.lower() == "healthy" and seg["percent"] > 5:
        st.warning("The model says healthy, but the segmentation found discoloured areas.")
    if not seg["reliable"]:
        st.warning("Very little leaf was detected â€” use a closer, clearer photo.")

    st.markdown("**Class probabilities**")
    ordered = sorted(result["scores"].items(), key=lambda item: item[1], reverse=True)
    for name, score in ordered:
        pretty = CLASS_INFO.get(name, (name.title(), ""))[0]
        st.progress(min(max(score, 0.0), 1.0), text=f"{pretty} â€” {score * 100:.1f}%")

st.divider()
st.subheader("Visual explanations")

tabs = ["Original", "Preprocessed", "Severity map", "Grad-CAM"]
tab_objects = st.tabs(tabs)

with tab_objects[0]:
    st.image(result["original"], caption="Input image", width="stretch")

with tab_objects[1]:
    st.image(
        result["processed"],
        caption="OpenCV cleaned image fed to the model (224 Ã— 224)",
        width="stretch",
    )
    if result["removed"] is not None:
        st.image(
            result["removed"],
            caption="Background removed before preprocessing",
            width="stretch",
        )

with tab_objects[2]:
    st.image(
        result["severity"],
        caption="Red = diseased tissue, green = healthy leaf, dimmed = background",
        width="stretch",
    )

with tab_objects[3]:
    st.image(
        result["gradcam"],
        caption="Warmer colours are the regions that most influenced the prediction",
        width="stretch",
    )

st.divider()
st.subheader("Download results")

d1, d2, d3 = st.columns(3)
d1.download_button(
    "Preprocessed image",
    data=encode_png(result["processed"]),
    file_name="preprocessed.png",
    mime="image/png",
    width="stretch",
)
d2.download_button(
    "Severity overlay",
    data=encode_png(result["severity"]),
    file_name="severity.png",
    mime="image/png",
    width="stretch",
)
d3.download_button(
    "Grad-CAM heatmap",
    data=encode_png(result["gradcam"]),
    file_name="gradcam.png",
    mime="image/png",
    width="stretch",
)

st.caption(
    "Pipeline: OpenCV preprocessing â†’ ResNet-50 classification â†’ HSV severity estimation â†’ Grad-CAM."
)

