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

# ── Session state — active page ────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Analysis"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header   { visibility: hidden; }

    .stApp { background-color: #f7f6f3; }

    /* ── nav bar shell ── */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 48px;
        height: 62px;
        border-bottom: 1px solid #e2e0db;
        background: #f7f6f3;
    }
    .nav-wordmark {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #1a1a1a;
    }
    .nav-wordmark span {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 0.72rem;
        color: #9a9a9a;
        margin-left: 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* ── nav buttons — Streamlit button overrides ── */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div > div > div > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #9a9a9a !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 6px 0 !important;
        min-height: unset !important;
        border-radius: 0 !important;
        width: auto !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div > div > div > button:hover {
        color: #1a1a1a !important;
        background: transparent !important;
    }

    /* ── hero ── */
    .hero {
        padding: 68px 48px 52px 48px;
        max-width: 820px;
    }
    .hero-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 18px;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.1rem;
        line-height: 1.13;
        color: #1a1a1a;
        margin-bottom: 22px;
        font-weight: 400;
    }
    .hero-sub {
        font-size: 1rem;
        color: #5a5a5a;
        line-height: 1.78;
        max-width: 580px;
    }

    /* ── page section ── */
    .page-header {
        padding: 40px 48px 0 48px;
    }
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.7rem;
        color: #1a1a1a;
        font-weight: 400;
        margin-bottom: 4px;
    }
    .page-rule {
        border: none;
        border-top: 1px solid #e2e0db;
        margin: 16px 0 32px 0;
    }

    /* ── disclaimer ── */
    .disclaimer {
        background: #f0ede8;
        border-radius: 3px;
        padding: 13px 18px;
        font-size: 0.77rem;
        color: #7a7a7a;
        line-height: 1.65;
        margin-bottom: 28px;
    }
    .disclaimer strong { color: #4a4a4a; }

    /* ── result card ── */
    .result-card {
        background: #fff;
        border: 1px solid #e2e0db;
        border-radius: 3px;
        padding: 26px 30px;
        margin: 20px 0 30px 0;
    }
    .result-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 10px;
    }
    .result-accent {
        width: 32px; height: 3px;
        background: #1a1a1a;
        border-radius: 2px;
        margin-bottom: 14px;
    }
    .result-class {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        color: #1a1a1a;
        margin-bottom: 4px;
        font-weight: 400;
    }
    .result-confidence {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 14px;
    }
    .result-desc {
        font-size: 0.86rem;
        color: #5a5a5a;
        line-height: 1.72;
        border-top: 1px solid #f0ede8;
        padding-top: 13px;
    }

    /* ── panel labels ── */
    .panel-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 10px;
    }
    .panel-caption {
        font-size: 0.74rem;
        color: #bbb;
        margin-top: 8px;
        line-height: 1.55;
    }

    /* ── probability rows ── */
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 0;
        border-bottom: 1px solid #f0ede8;
    }
    .prob-row:last-child { border-bottom: none; }
    .prob-name          { font-size: 0.83rem; color: #4a4a4a; }
    .prob-name.active   { font-weight: 600; color: #1a1a1a; }
    .prob-val           { font-size: 0.83rem; color: #888; font-variant-numeric: tabular-nums; }
    .prob-val.active    { color: #1a1a1a; font-weight: 600; }
    .prob-bar-bg        { width: 100%; height: 2px; background: #ece9e4; border-radius: 2px; margin-top: 4px; }
    .prob-bar-fill      { height: 2px; background: #1a1a1a; border-radius: 2px; }

    /* ── report panels ── */
    .report-section-sub {
        font-size: 0.83rem;
        color: #7a7a7a;
        line-height: 1.65;
        margin-bottom: 30px;
        max-width: 640px;
    }
    .report-panel-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e0db;
    }

    /* ── about section ── */
    .about-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-top: 8px;
    }
    .about-card {
        background: #fff;
        border: 1px solid #e2e0db;
        border-radius: 3px;
        padding: 26px 28px;
    }
    .about-card-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 10px;
    }
    .about-card-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.15rem;
        color: #1a1a1a;
        font-weight: 400;
        margin-bottom: 10px;
    }
    .about-card-body {
        font-size: 0.84rem;
        color: #5a5a5a;
        line-height: 1.75;
    }
    .about-card-body ul {
        padding-left: 16px;
        margin: 0;
    }
    .about-card-body li { margin-bottom: 5px; }
    .about-stat {
        display: flex;
        gap: 40px;
        margin-top: 28px;
    }
    .about-stat-item { }
    .about-stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #1a1a1a;
        line-height: 1;
        margin-bottom: 4px;
    }
    .about-stat-desc {
        font-size: 0.74rem;
        color: #aaa;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .about-full-width {
        grid-column: 1 / -1;
    }

    /* ── empty state ── */
    .empty-state {
        text-align: center;
        padding: 80px 32px;
    }
    .empty-state-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.35rem;
        color: #c8c5be;
        font-weight: 400;
        margin-bottom: 8px;
    }
    .empty-state-sub { font-size: 0.8rem; color: #c8c5be; }

    /* ── file uploader ── */
    [data-testid="stFileUploaderDropzone"] {
        background: #faf9f7 !important;
        border: 1.5px dashed #c8c5be !important;
        border-radius: 3px !important;
    }

    /* ── active nav indicator (injected via JS-free trick) ── */
    .nav-active {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a;
        padding-bottom: 2px;
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
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    tensor = tf.convert_to_tensor(image_4d, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tensor)
        pred_cls = tf.argmax(preds[0])
        score = preds[:, pred_cls]
    grads  = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm     = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    hm     = tf.maximum(hm, 0)
    mx     = tf.reduce_max(hm)
    hm     = tf.cond(mx > 0, lambda: hm / mx, lambda: hm)
    return hm.numpy(), int(pred_cls)

def overlay_gradcam(gray_arr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    H, W = gray_arr.shape
    hm_r = (
        np.asarray(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
        ).astype(np.float32) / 255.0
    )
    colored = cm.get_cmap("inferno")(hm_r)[:, :, :3]
    base    = np.stack([gray_arr] * 3, axis=-1)
    return (np.clip((1 - alpha) * base + alpha * colored, 0, 1) * 255).astype(np.uint8)

# ── Confidence chart ───────────────────────────────────────────────────────────
def confidence_chart(probs: np.ndarray) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    colors = ["#1a1a1a" if i == int(np.argmax(probs)) else "#e2e0db" for i in range(len(CLASS_NAMES))]
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor="none", height=0.45)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=8, color="#9a9a9a")
    ax.tick_params(labelsize=8, colors="#4a4a4a", length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, color="#c8c5be")
    ax.set_axisbelow(True)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{p * 100:.1f}%", va="center", fontsize=8, color="#4a4a4a")
    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR  — wordmark left, three real buttons right
# ══════════════════════════════════════════════════════════════════════════════
nav_left, nav_right = st.columns([6, 1.8])

with nav_left:
    st.markdown(
        "<div style='padding:18px 0 14px 0; font-family:\"DM Serif Display\",serif; font-size:1.1rem; color:#1a1a1a;'>"
        "NeuroScan <span style='font-family:Inter,sans-serif;font-size:0.72rem;color:#aaa;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-left:8px;'>Research Tool</span></div>",
        unsafe_allow_html=True,
    )

with nav_right:
    nb1, nb2, nb3 = st.columns(3)
    with nb1:
        if st.button("Analysis"):
            st.session_state.page = "Analysis"
    with nb2:
        if st.button("Reports"):
            st.session_state.page = "Reports"
    with nb3:
        if st.button("About"):
            st.session_state.page = "About"

st.markdown("<hr style='border:none;border-top:1px solid #e2e0db;margin:0 0 0 0;'>", unsafe_allow_html=True)

# ── active page indicator strip ───────────────────────────────────────────────
page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# HERO  (shown on Analysis and About pages)
# ══════════════════════════════════════════════════════════════════════════════
if page in ("Analysis", "About"):
    hero_sub = (
        "A convolutional neural network trained to classify brain MRI scans "
        "into three tumor categories. Upload a scan to receive a classification "
        "with gradient-weighted activation mapping."
        if page == "Analysis"
        else
        "Details about the dataset, model architecture, and the team behind this research tool."
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-label">Deep Learning &nbsp;&middot;&nbsp; MRI Classification</div>
            <div class="hero-title">Brain Tumor<br>MRI Analysis</div>
            <div class="hero-sub">{hero_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Analysis":
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

    model = load_model()
    uploaded = st.file_uploader(
        "Select a brain MRI image (PNG, JPG, TIFF)",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        tensor  = preprocess_image(pil_img)

        with st.spinner("Running inference…"):
            probs      = model.predict(tensor, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        with st.spinner("Computing activation map…"):
            heatmap, _ = make_gradcam_heatmap(tensor, model, LAST_CONV_LAYER)
            gray_arr   = tensor[0, :, :, 0]
            overlay    = overlay_gradcam(gray_arr, heatmap)

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
            prob_html = ""
            for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
                ac = "active" if i == pred_idx else ""
                prob_html += f"""
                <div class="prob-row">
                    <div style="flex:1">
                        <div class="prob-name {ac}">{name}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{p*100:.1f}%"></div>
                        </div>
                    </div>
                    <div class="prob-val {ac}" style="margin-left:16px">{p*100:.2f}%</div>
                </div>"""
            st.markdown(prob_html, unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No image selected</div>
                <div class="empty-state-sub">Upload a brain MRI scan above to begin analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Model Performance</div>
            <div class="hero-title">Training Reports</div>
            <div class="hero-sub">
                Performance metrics from the final model, trained on 3,064 MRI images
                across 233 patients using a strict patient-level train / validation / test
                split to prevent data leakage.
            </div>
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown(
        """
        <div class="about-grid">

            <div class="about-card">
                <div class="about-card-label">The Project</div>
                <div class="about-card-title">What this tool does</div>
                <div class="about-card-body">
                    This application uses a custom convolutional neural network to classify
                    brain MRI scans into three tumor categories: Meningioma, Glioma, and
                    Pituitary Tumor. Alongside each classification, a Gradient-weighted Class
                    Activation Map (Grad-CAM) is generated to highlight the image regions that
                    most influenced the model's decision.
                    <div class="about-stat">
                        <div class="about-stat-item">
                            <div class="about-stat-num">3,064</div>
                            <div class="about-stat-desc">Training images</div>
                        </div>
                        <div class="about-stat-item">
                            <div class="about-stat-num">233</div>
                            <div class="about-stat-desc">Patients</div>
                        </div>
                        <div class="about-stat-item">
                            <div class="about-stat-num">3</div>
                            <div class="about-stat-desc">Tumor classes</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="about-card">
                <div class="about-card-label">Dataset</div>
                <div class="about-card-title">Data source &amp; split strategy</div>
                <div class="about-card-body">
                    The model was trained on a publicly available brain MRI dataset.
                    A strict <strong>patient-level split</strong> was applied — images from the
                    same patient appear in only one partition — to prevent data leakage and
                    ensure the evaluation reflects real-world generalisation.
                    <ul style="margin-top:12px">
                        <li>Images resized to 224 × 224 px (grayscale)</li>
                        <li>Per-image min-max normalisation to [0, 1]</li>
                        <li>Train / Validation / Test: 70 / 15 / 15 %</li>
                    </ul>
                </div>
            </div>

            <div class="about-card">
                <div class="about-card-label">Architecture</div>
                <div class="about-card-title">Model design</div>
                <div class="about-card-body">
                    The classifier is a custom CNN built from scratch — no pretrained weights.
                    <ul style="margin-top:12px">
                        <li>6 convolutional blocks with batch normalisation and max pooling</li>
                        <li>Global average pooling into two dense layers</li>
                        <li>Dropout for regularisation</li>
                        <li>Softmax output over 3 classes</li>
                        <li>Trained with Adam optimiser and early stopping (19 epochs)</li>
                    </ul>
                </div>
            </div>

            <div class="about-card">
                <div class="about-card-label">Explainability</div>
                <div class="about-card-title">Grad-CAM visualisation</div>
                <div class="about-card-body">
                    Gradient-weighted Class Activation Mapping (Grad-CAM) backpropagates
                    the gradient of the predicted class score through the final convolutional
                    layer (<code>conv2d_5</code>) to produce a spatial heatmap. Brighter
                    regions in the overlay correspond to areas the model found most
                    discriminative for its prediction. This aids interpretability without
                    modifying the underlying model.
                </div>
            </div>

            <div class="about-card about-full-width">
                <div class="about-card-label">Disclaimer</div>
                <div class="about-card-title">Research use only</div>
                <div class="about-card-body">
                    This tool is developed strictly for educational and research purposes.
                    It has not been clinically validated and must not be used to inform,
                    support, or replace any medical diagnosis or treatment decision.
                    Always consult a qualified medical professional for clinical concerns.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )
