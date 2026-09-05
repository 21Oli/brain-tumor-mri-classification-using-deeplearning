import os
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "models", "best_brain_tumor_cnn.keras")
REPORTS_DIR     = os.path.join(os.path.dirname(__file__), "reports")
CLASS_NAMES     = ["Meningioma", "Glioma", "Pituitary Tumor"]
IMG_SIZE        = 224
LAST_CONV_LAYER = "conv2d_5"

CLASS_INFO = {
    "Meningioma": {
        "color": "#f0a500",
        "description": (
            "Meningiomas arise from the meninges, the membranes surrounding the brain "
            "and spinal cord. Most are benign and slow-growing."
        ),
    },
    "Glioma": {
        "color": "#e05252",
        "description": (
            "Gliomas develop from glial cells within the brain. They are the most common "
            "primary brain tumors and include glioblastomas."
        ),
    },
    "Pituitary Tumor": {
        "color": "#4a90d9",
        "description": (
            "Pituitary tumors form in the pituitary gland. Most are non-cancerous "
            "(adenomas) and can affect hormone regulation."
        ),
    },
}


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Matches the training pipeline exactly:
      1. Grayscale
      2. Resize 224×224 (bilinear)
      3. float32 + per-image min-max normalise → [0, 1]
      4. Expand to (1, 224, 224, 1)
    """
    img = pil_image.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return arr[..., np.newaxis][np.newaxis, ...]   # (1, 224, 224, 1)


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(
    image_4d: np.ndarray, model, last_conv_layer_name: str
):
    """Returns a (H, W) float32 heatmap in [0, 1] and the predicted class index."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )
    tensor = tf.convert_to_tensor(image_4d, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tensor)
        pred_cls = tf.argmax(preds[0])
        score = preds[:, pred_cls]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    mx = tf.reduce_max(heatmap)
    heatmap = tf.cond(mx > 0, lambda: heatmap / mx, lambda: heatmap)
    return heatmap.numpy(), int(pred_cls)


def overlay_gradcam(
    gray_arr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """
    Superimposes a Grad-CAM heatmap (jet colormap) over a grayscale MRI.
    gray_arr : (H, W) float32 [0, 1]
    heatmap  : (h, w) float32 [0, 1]
    Returns  : (H, W, 3) uint8
    """
    H, W = gray_arr.shape
    hm_resized = (
        np.asarray(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (W, H), Image.Resampling.BILINEAR
            )
        ).astype(np.float32)
        / 255.0
    )
    colored = cm.get_cmap("jet")(hm_resized)[:, :, :3]
    base_rgb = np.stack([gray_arr] * 3, axis=-1)
    out = np.clip((1 - alpha) * base_rgb + alpha * colored, 0, 1)
    return (out * 255).astype(np.uint8)


# ── Confidence bar chart ───────────────────────────────────────────────────────
def confidence_chart(probs: np.ndarray) -> io.BytesIO:
    colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%",
            va="center",
            fontsize=8.5,
        )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center;margin-bottom:0'>🧠 Brain Tumor MRI Classifier</h1>
    <p style='text-align:center;color:#888;margin-top:4px'>
        Custom CNN &nbsp;·&nbsp; 3-class &nbsp;·&nbsp;
        224×224 grayscale &nbsp;·&nbsp; Test accuracy 70.76 %
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

tab_predict, tab_reports = st.tabs(["🔬 Classify MRI", "📊 Model Reports"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    model = load_model()

    st.markdown(
        "**Upload a brain MRI scan** (PNG, JPG, or TIFF) to classify it as "
        "*Meningioma*, *Glioma*, or *Pituitary Tumor*. "
        "A Grad-CAM overlay highlights the regions that most influenced the prediction.\n\n"
        "> ⚠️ **Disclaimer:** This is an educational ML project. "
        "It is **not** a clinical diagnostic tool and must not be used for medical decisions."
    )

    uploaded = st.file_uploader(
        "Upload MRI image",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        tensor = preprocess_image(pil_img)

        with st.spinner("Running inference…"):
            probs = model.predict(tensor, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        with st.spinner("Generating Grad-CAM…"):
            heatmap, _ = make_gradcam_heatmap(tensor, model, LAST_CONV_LAYER)
            gray_arr = tensor[0, :, :, 0]           # (224, 224) float32
            overlay = overlay_gradcam(gray_arr, heatmap)

        # Result banner
        color = CLASS_INFO[pred_class]["color"]
        st.markdown(
            f"""
            <div style='
                background:{color}18;
                border-left:5px solid {color};
                border-radius:6px;
                padding:14px 20px;
                margin:16px 0 8px 0;
            '>
                <span style='font-size:1.5rem;font-weight:700;color:{color}'>
                    {pred_class}
                </span>
                <span style='font-size:1.1rem;color:#555;margin-left:14px'>
                    {confidence * 100:.1f}% confidence
                </span>
                <p style='margin:8px 0 0 0;color:#555;font-size:.92rem'>
                    {CLASS_INFO[pred_class]['description']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1, 1, 1.1])

        with c1:
            st.markdown("**Input MRI**")
            st.image(pil_img.convert("L"), use_container_width=True, clamp=True)

        with c2:
            st.markdown("**Grad-CAM Overlay**")
            st.image(overlay, use_container_width=True, clamp=True)
            st.caption("Warmer regions (red/yellow) most influenced the prediction.")

        with c3:
            st.markdown("**Prediction Confidence**")
            st.image(confidence_chart(probs), use_container_width=True)
            st.markdown("**All class probabilities:**")
            for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
                marker = "✅" if i == pred_idx else "　"
                st.write(f"{marker} {name}: **{p * 100:.2f}%**")

    else:
        st.info("Upload a brain MRI image above to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_reports:
    st.subheader("Training Reports")
    st.markdown(
        "Results from the final model trained on 3,064 MRI images "
        "(233 patients, patient-level train/val/test split)."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Training vs Validation Accuracy**")
        p = os.path.join(REPORTS_DIR, "training_accuracy.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("training_accuracy.png not found.")

    with c2:
        st.markdown("**Training vs Validation Loss**")
        p = os.path.join(REPORTS_DIR, "training_loss.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("training_loss.png not found.")

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Confusion Matrix (Test Set)**")
        p = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found.")

    with c4:
        st.markdown("**ROC-AUC Curves (One-vs-Rest)**")
        p = os.path.join(REPORTS_DIR, "roc_auc.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("roc_auc.png not found.")

    st.markdown("---")
    st.markdown("**Classification Report**")
    p = os.path.join(REPORTS_DIR, "classification_report.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            st.code(f.read(), language=None)
    else:
        st.warning("classification_report.txt not found.")
