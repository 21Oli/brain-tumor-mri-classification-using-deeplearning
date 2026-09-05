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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- fonts & base ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- page background ---- */
    .stApp {
        background-color: #f7f6f3;
    }

    /* ---- top nav bar ---- */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 22px 48px 18px 48px;
        border-bottom: 1px solid #e2e0db;
        background: #f7f6f3;
        margin-bottom: 0;
    }
    .nav-wordmark {
        font-family: 'DM Serif Display', serif;
        font-size: 1.15rem;
        color: #1a1a1a;
        letter-spacing: 0.01em;
    }
    .nav-wordmark span {
        color: #6b6b6b;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 0.78rem;
        margin-left: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .nav-links {
        display: flex;
        gap: 32px;
    }
    .nav-links a {
        font-size: 0.82rem;
        font-weight: 500;
        color: #4a4a4a;
        text-decoration: none;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---- hero ---- */
    .hero {
        padding: 72px 48px 56px 48px;
        max-width: 820px;
    }
    .hero-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #9a9a9a;
        margin-bottom: 18px;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        line-height: 1.15;
        color: #1a1a1a;
        margin-bottom: 22px;
        font-weight: 400;
    }
    .hero-sub {
        font-size: 1.02rem;
        color: #5a5a5a;
        line-height: 1.75;
        max-width: 600px;
        font-weight: 400;
    }

    /* ---- divider ---- */
    .section-divider {
        border: none;
        border-top: 1px solid #e2e0db;
        margin: 0 0 40px 0;
    }

    /* ---- tab overrides ---- */
    div[data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #e2e0db !important;
        gap: 0 !important;
        padding: 0 48px !important;
    }
    div[data-baseweb="tab"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #9a9a9a !important;
        padding: 14px 24px !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a !important;
    }
    div[data-baseweb="tab-panel"] {
        padding: 40px 48px !important;
    }

    /* ---- upload zone ---- */
    .upload-zone {
        border: 1.5px dashed #c8c5be;
        border-radius: 4px;
        background: #faf9f7;
        padding: 48px 32px;
        text-align: center;
        margin-bottom: 32px;
    }
    .upload-zone-title {
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #3a3a3a;
        margin-bottom: 8px;
    }
    .upload-zone-sub {
        font-size: 0.8rem;
        color: #9a9a9a;
    }

    /* ---- result card ---- */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e0db;
        border-radius: 4px;
        padding: 28px 32px;
        margin: 24px 0 32px 0;
    }
    .result-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #9a9a9a;
        margin-bottom: 10px;
    }
    .result-class {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #1a1a1a;
        margin-bottom: 6px;
        font-weight: 400;
    }
    .result-confidence {
        font-size: 0.88rem;
        color: #6b6b6b;
        margin-bottom: 14px;
        font-weight: 400;
    }
    .result-desc {
        font-size: 0.88rem;
        color: #5a5a5a;
        line-height: 1.7;
        border-top: 1px solid #f0ede8;
        padding-top: 14px;
        margin-top: 4px;
    }
    .result-accent {
        width: 36px;
        height: 3px;
        background: #1a1a1a;
        margin-bottom: 16px;
        border-radius: 2px;
    }

    /* ---- image panels ---- */
    .panel-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9a9a9a;
        margin-bottom: 10px;
    }
    .panel-caption {
        font-size: 0.76rem;
        color: #aaa;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* ---- probability rows ---- */
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f0ede8;
    }
    .prob-row:last-child { border-bottom: none; }
    .prob-name {
        font-size: 0.84rem;
        color: #3a3a3a;
        font-weight: 400;
    }
    .prob-name.active {
        font-weight: 600;
        color: #1a1a1a;
    }
    .prob-val {
        font-size: 0.84rem;
        color: #6b6b6b;
        font-variant-numeric: tabular-nums;
    }
    .prob-val.active {
        color: #1a1a1a;
        font-weight: 600;
    }
    .prob-bar-bg {
        width: 100%;
        height: 3px;
        background: #f0ede8;
        border-radius: 2px;
        margin-top: 5px;
    }
    .prob-bar-fill {
        height: 3px;
        background: #1a1a1a;
        border-radius: 2px;
    }

    /* ---- disclaimer ---- */
    .disclaimer {
        background: #f0ede8;
        border-radius: 4px;
        padding: 14px 20px;
        font-size: 0.78rem;
        color: #7a7a7a;
        line-height: 1.6;
        margin-bottom: 32px;
    }
    .disclaimer strong { color: #4a4a4a; }

    /* ---- report section labels ---- */
    .report-section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: #1a1a1a;
        font-weight: 400;
        margin-bottom: 6px;
    }
    .report-section-sub {
        font-size: 0.84rem;
        color: #7a7a7a;
        margin-bottom: 32px;
        line-height: 1.6;
    }
    .report-panel-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9a9a9a;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e0db;
    }

    /* ---- info state ---- */
    .empty-state {
        text-align: center;
        padding: 80px 32px;
        color: #aaa;
    }
    .empty-state-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #c8c5be;
        font-weight: 400;
        margin-bottom: 10px;
    }
    .empty-state-sub {
        font-size: 0.82rem;
        color: #bbb;
    }

    /* ---- streamlit file uploader tweaks ---- */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #faf9f7 !important;
        border: 1.5px dashed #c8c5be !important;
        border-radius: 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "models", "best_brain_tumor_cnn.keras")
REPORTS_DIR     = os.path.join(os.path.dirname(__file__), "reports")
CLASS_NAMES     = ["Meningioma", "Glioma", "Pituitary Tumor"]
IMG_SIZE        = 224
LAST_CONV_LAYER = "conv2d_5"

CLASS_INFO = {
    "Meningioma": {
        "description": (
            "Meningiomas originate from the meninges — the protective membranes "
            "enveloping the brain and spinal cord. The majority are benign and "
            "progress slowly over time."
        ),
    },
    "Glioma": {
        "description": (
            "Gliomas arise from the glial support cells of the brain and represent "
            "the most prevalent category of primary brain tumors, encompassing "
            "grades from low-grade astrocytomas to glioblastoma multiforme."
        ),
    },
    "Pituitary Tumor": {
        "description": (
            "Pituitary tumors develop within the pituitary gland at the base of "
            "the skull. Most are non-cancerous adenomas that may disrupt hormonal "
            "regulation and adjacent neural structures."
        ),
    },
}


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising model…")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return arr[..., np.newaxis][np.newaxis, ...]


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(image_4d: np.ndarray, model, last_conv_layer_name: str):
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


def overlay_gradcam(gray_arr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    H, W = gray_arr.shape
    hm_resized = (
        np.asarray(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (W, H), Image.Resampling.BILINEAR
            )
        ).astype(np.float32) / 255.0
    )
    colored = cm.get_cmap("inferno")(hm_resized)[:, :, :3]
    base_rgb = np.stack([gray_arr] * 3, axis=-1)
    out = np.clip((1 - alpha) * base_rgb + alpha * colored, 0, 1)
    return (out * 255).astype(np.uint8)


# ── Confidence chart (minimal, matches UI palette) ────────────────────────────
def confidence_chart(probs: np.ndarray) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bar_colors = ["#1a1a1a" if i == int(np.argmax(probs)) else "#e2e0db"
                  for i in range(len(CLASS_NAMES))]
    bars = ax.barh(CLASS_NAMES, probs * 100, color=bar_colors, edgecolor="none", height=0.45)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=8, color="#9a9a9a")
    ax.tick_params(labelsize=8, colors="#4a4a4a")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, color="#c8c5be")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%",
            va="center",
            fontsize=8,
            color="#4a4a4a",
        )

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-wordmark">
            NeuroScan <span>Research Tool</span>
        </div>
        <div class="nav-links">
            <a href="#">Analysis</a>
            <a href="#">Reports</a>
            <a href="#">About</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Deep Learning &nbsp;&middot;&nbsp; MRI Classification</div>
        <div class="hero-title">Brain Tumor<br>MRI Analysis</div>
        <div class="hero-sub">
            A convolutional neural network trained to classify brain MRI scans
            into three tumor categories. Upload a scan to receive a classification
            with gradient-weighted activation mapping.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_reports = st.tabs(["Analysis", "Model Reports"])


# ── TAB 1  ────────────────────────────────────────────────────────────────────
with tab_predict:
    model = load_model()

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Research use only.</strong> This tool is built for educational and
            research purposes. It is not validated for clinical diagnosis and must
            not inform any medical decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Select a brain MRI image (PNG, JPG, TIFF)",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        tensor = preprocess_image(pil_img)

        with st.spinner("Running inference…"):
            probs = model.predict(tensor, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        with st.spinner("Computing activation map…"):
            heatmap, _ = make_gradcam_heatmap(tensor, model, LAST_CONV_LAYER)
            gray_arr = tensor[0, :, :, 0]
            overlay = overlay_gradcam(gray_arr, heatmap)

        # Result card
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Classification Result</div>
                <div class="result-accent"></div>
                <div class="result-class">{pred_class}</div>
                <div class="result-confidence">{confidence * 100:.1f}% model confidence</div>
                <div class="result-desc">{CLASS_INFO[pred_class]['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Three columns
        c1, c2, c3 = st.columns([1, 1, 1], gap="large")

        with c1:
            st.markdown('<div class="panel-label">Input Scan</div>', unsafe_allow_html=True)
            st.image(pil_img.convert("L"), use_container_width=True, clamp=True)

        with c2:
            st.markdown('<div class="panel-label">Gradient Activation Map</div>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True, clamp=True)
            st.markdown(
                '<div class="panel-caption">Brighter regions indicate areas the model weighted most heavily when forming its prediction.</div>',
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown('<div class="panel-label">Class Probabilities</div>', unsafe_allow_html=True)
            st.image(confidence_chart(probs), use_container_width=True)

            # Probability list
            prob_rows_html = ""
            for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
                active_cls = "active" if i == pred_idx else ""
                fill_w = f"{p * 100:.1f}"
                prob_rows_html += f"""
                <div class="prob-row">
                    <div>
                        <div class="prob-name {active_cls}">{name}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{fill_w}%"></div>
                        </div>
                    </div>
                    <div class="prob-val {active_cls}">{p * 100:.2f}%</div>
                </div>
                """
            st.markdown(prob_rows_html, unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No image selected</div>
                <div class="empty-state-sub">
                    Upload a brain MRI scan above to begin analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── TAB 2  ────────────────────────────────────────────────────────────────────
with tab_reports:
    st.markdown(
        """
        <div class="report-section-title">Training Reports</div>
        <div class="report-section-sub">
            Performance metrics from the final model, trained on 3,064 MRI images
            across 233 patients using a strict patient-level train / validation / test split
            to prevent data leakage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="report-panel-label">Training vs Validation Accuracy</div>', unsafe_allow_html=True)
        p = os.path.join(REPORTS_DIR, "training_accuracy.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("training_accuracy.png not found.")

    with c2:
        st.markdown('<div class="report-panel-label">Training vs Validation Loss</div>', unsafe_allow_html=True)
        p = os.path.join(REPORTS_DIR, "training_loss.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("training_loss.png not found.")

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown('<div class="report-panel-label">Confusion Matrix — Test Set</div>', unsafe_allow_html=True)
        p = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found.")

    with c4:
        st.markdown('<div class="report-panel-label">ROC-AUC Curves — One vs Rest</div>', unsafe_allow_html=True)
        p = os.path.join(REPORTS_DIR, "roc_auc.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning("roc_auc.png not found.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="report-panel-label">Classification Report</div>', unsafe_allow_html=True)
    p = os.path.join(REPORTS_DIR, "classification_report.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            st.code(f.read(), language=None)
    else:
        st.warning("classification_report.txt not found.")
