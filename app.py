
import streamlit as st
import numpy as np
import cv2
import joblib
import time

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Gender Classifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("modell.pkl")
IMG_SIZE = 64

# -------------------------
# Helper: Circular Confidence Ring
# -------------------------
def confidence_ring(label, percent, color1="#7c3aed", color2="#06b6d4"):
    return f"""
    <div class="ring-card">
        <div class="ring-wrap" style="--percent:{percent}; --c1:{color1}; --c2:{color2};">
            <div class="ring-inner">
                <div class="ring-value">{percent:.1f}%</div>
            </div>
        </div>
        <div class="ring-label">{label}</div>
    </div>
    """

# -------------------------
# Premium CSS
# -------------------------
st.markdown("""
<style>
/* =========================
   GLOBAL BACKGROUND
========================= */
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(124, 58, 237, 0.28), transparent 28%),
        radial-gradient(circle at 85% 25%, rgba(6, 182, 212, 0.24), transparent 28%),
        radial-gradient(circle at 50% 85%, rgba(34, 197, 94, 0.18), transparent 30%),
        linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
    color: white;
    overflow-x: hidden;
}

/* Aurora overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: -10%;
    background:
        radial-gradient(circle at 20% 30%, rgba(168, 85, 247, 0.18), transparent 22%),
        radial-gradient(circle at 80% 20%, rgba(34, 211, 238, 0.15), transparent 22%),
        radial-gradient(circle at 55% 80%, rgba(74, 222, 128, 0.12), transparent 24%);
    filter: blur(55px);
    animation: auroraMove 15s ease-in-out infinite alternate;
    z-index: -3;
}
@keyframes auroraMove {
    0%   { transform: translate3d(0, 0, 0) scale(1); }
    50%  { transform: translate3d(25px, -18px, 0) scale(1.04); }
    100% { transform: translate3d(-18px, 18px, 0) scale(1.02); }
}

/* =========================
   FLOATING PARTICLES
========================= */
.particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: -2;
}
.particle {
    position: absolute;
    border-radius: 50%;
    background: rgba(255,255,255,0.12);
    box-shadow: 0 0 18px rgba(255,255,255,0.16);
    animation: drift linear infinite;
}
.p1 { top: 12%; left: 14%; width: 8px; height: 8px; animation-duration: 13s; }
.p2 { top: 25%; left: 80%; width: 12px; height: 12px; animation-duration: 16s; }
.p3 { top: 68%; left: 20%; width: 9px; height: 9px; animation-duration: 18s; }
.p4 { top: 76%; left: 72%; width: 7px; height: 7px; animation-duration: 14s; }
.p5 { top: 45%; left: 50%; width: 10px; height: 10px; animation-duration: 20s; }
.p6 { top: 15%; left: 55%; width: 6px; height: 6px; animation-duration: 17s; }
.p7 { top: 58%; left: 88%; width: 11px; height: 11px; animation-duration: 15s; }
.p8 { top: 84%; left: 42%; width: 8px; height: 8px; animation-duration: 19s; }

@keyframes drift {
    0%   { transform: translateY(0px) translateX(0px); opacity: 0.2; }
    25%  { opacity: 0.55; }
    50%  { transform: translateY(-28px) translateX(18px); opacity: 0.3; }
    75%  { opacity: 0.6; }
    100% { transform: translateY(0px) translateX(0px); opacity: 0.2; }
}

/* =========================
   SIDEBAR
========================= */
section[data-testid="stSidebar"] {
    background: rgba(10, 18, 33, 0.92);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.sidebar-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}
.sidebar-text {
    color: #dbeafe;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* =========================
   HERO
========================= */
.hero {
    position: relative;
    overflow: hidden;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(16px);
    border-radius: 30px;
    padding: 34px;
    text-align: center;
    box-shadow: 0 12px 34px rgba(0,0,0,0.28);
    margin-bottom: 26px;
    animation: fadeUp 0.9s ease;
}
.hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: -120%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    transform: skewX(-20deg);
    animation: heroShine 4.5s linear infinite;
}
@keyframes heroShine {
    0%   { left: -120%; }
    100% { left: 150%; }
}
.hero h1 {
    margin: 0;
    font-size: 3rem;
    color: #ffffff;
    text-shadow: 0 0 18px rgba(255,255,255,0.10);
}
.hero p {
    margin-top: 10px;
    color: #dbeafe;
    font-size: 1.08rem;
}

/* =========================
   GLASS CARDS
========================= */
.glass-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    transition: transform 0.35s ease, box-shadow 0.35s ease;
    animation: fadeUp 0.9s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(0,0,0,0.30);
}

/* =========================
   UPLOADER
========================= */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 2px dashed rgba(255,255,255,0.25);
    border-radius: 18px;
    padding: 12px;
    box-shadow: inset 0 0 18px rgba(255,255,255,0.04);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
section[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.85);
    box-shadow: 0 0 22px rgba(96,165,250,0.18);
}

/* =========================
   IMAGE FRAME
========================= */
.image-frame {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 3px;
    background: linear-gradient(135deg, rgba(168,85,247,0.8), rgba(34,211,238,0.8), rgba(74,222,128,0.8));
    box-shadow: 0 0 24px rgba(34,211,238,0.15);
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.image-frame:hover {
    transform: scale(1.015);
    box-shadow: 0 0 34px rgba(34,211,238,0.24);
}
.image-inner {
    background: rgba(7, 12, 22, 0.88);
    border-radius: 22px;
    padding: 6px;
}

/* =========================
   PREDICTION CARD
========================= */
.prediction-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(34,197,94,0.88), rgba(16,185,129,0.80));
    border-radius: 26px;
    padding: 28px;
    text-align: center;
    color: white;
    box-shadow: 0 12px 34px rgba(16,185,129,0.30);
    animation: pulseGlow 2.6s ease-in-out infinite;
    margin-bottom: 18px;
}
.prediction-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: -130%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent);
    transform: skewX(-20deg);
    animation: shimmer 3.8s linear infinite;
}
@keyframes shimmer {
    0%   { left: -130%; }
    100% { left: 150%; }
}
@keyframes pulseGlow {
    0%   { box-shadow: 0 0 0 rgba(16,185,129,0.25); }
    50%  { box-shadow: 0 0 34px rgba(16,185,129,0.45); }
    100% { box-shadow: 0 0 0 rgba(16,185,129,0.25); }
}
.prediction-text {
    font-size: 2rem;
    font-weight: 800;
}
.prediction-sub {
    margin-top: 8px;
    color: #ecfdf5;
}

/* =========================
   PROBABILITY BOXES
========================= */
.prob-box {
    background: rgba(255,255,255,0.11);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.prob-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(0,0,0,0.18);
}
.metric-title {
    color: #e5f4ff;
    font-weight: 600;
    font-size: 0.98rem;
}
.metric-value {
    color: #ffffff;
    font-weight: 800;
    font-size: 1.35rem;
}

/* =========================
   CIRCULAR CONFIDENCE RING
========================= */
.ring-card {
    text-align: center;
    animation: fadeUp 1s ease;
}
.ring-wrap {
    --size: 180px;
    width: var(--size);
    height: var(--size);
    margin: 0 auto 14px auto;
    border-radius: 50%;
    background:
        conic-gradient(from 0deg, var(--c1) 0%, var(--c2) calc(var(--percent) * 1%), rgba(255,255,255,0.08) 0);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 28px rgba(124,58,237,0.22);
    position: relative;
}
.ring-wrap::after {
    content: "";
    position: absolute;
    inset: 10px;
    border-radius: 50%;
    background: rgba(10, 16, 28, 0.95);
    box-shadow: inset 0 0 18px rgba(255,255,255,0.04);
}
.ring-inner {
    position: relative;
    z-index: 2;
    text-align: center;
}
.ring-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: white;
}
.ring-label {
    font-size: 1rem;
    color: #dbeafe;
    font-weight: 600;
}

/* =========================
   SUMMARY CARDS
========================= */
.summary-card {
    text-align: center;
    padding: 18px;
    border-radius: 20px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.20);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeUp 1s ease;
}
.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 34px rgba(0,0,0,0.28);
}

/* =========================
   BADGE
========================= */
.badge {
    display: inline-block;
    padding: 10px 16px;
    border-radius: 999px;
    background: rgba(34,197,94,0.16);
    border: 1px solid rgba(34,197,94,0.25);
    color: #dcfce7;
    font-weight: 700;
    box-shadow: 0 0 18px rgba(34,197,94,0.10);
}

/* =========================
   PROGRESS BAR
========================= */
div[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e) !important;
    border-radius: 999px !important;
}

/* =========================
   FLYING HEARTS EFFECT
========================= */
.hearts-container {
    position: fixed;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
    z-index: 9999;
}
.heart {
    position: absolute;
    bottom: -50px;
    font-size: 26px;
    opacity: 0;
    animation: floatHeart linear forwards;
    filter: drop-shadow(0 0 8px rgba(255,255,255,0.25));
}
@keyframes floatHeart {
    0% {
        transform: translateY(0) scale(0.7) rotate(0deg);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    50% {
        transform: translateY(-45vh) translateX(20px) scale(1.1) rotate(180deg);
        opacity: 0.95;
    }
    100% {
        transform: translateY(-110vh) translateX(-20px) scale(1.45) rotate(360deg);
        opacity: 0;
    }
}

.h1  { left: 5%;  color: #ff4d6d; animation-duration: 7s;  font-size: 22px; }
.h2  { left: 12%; color: #ff85a1; animation-duration: 9s;  font-size: 28px; }
.h3  { left: 20%; color: #ff006e; animation-duration: 8s;  font-size: 20px; }
.h4  { left: 28%; color: #fb6f92; animation-duration: 10s; font-size: 30px; }
.h5  { left: 36%; color: #c77dff; animation-duration: 7.5s; font-size: 24px; }
.h6  { left: 44%; color: #9d4edd; animation-duration: 9.5s; font-size: 26px; }
.h7  { left: 52%; color: #ff758f; animation-duration: 8.5s; font-size: 22px; }
.h8  { left: 60%; color: #ff4d6d; animation-duration: 11s; font-size: 32px; }
.h9  { left: 68%; color: #ff8fab; animation-duration: 8.8s; font-size: 25px; }
.h10 { left: 76%; color: #f72585; animation-duration: 10.5s; font-size: 29px; }
.h11 { left: 84%; color: #b5179e; animation-duration: 9.2s; font-size: 24px; }
.h12 { left: 92%; color: #ff5d8f; animation-duration: 11.5s; font-size: 30px; }

/* =========================
   FOOTER + ANIMATION
========================= */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 28px;
    font-size: 0.95rem;
}
</style>

<div class="particles">
    <div class="particle p1"></div>
    <div class="particle p2"></div>
    <div class="particle p3"></div>
    <div class="particle p4"></div>
    <div class="particle p5"></div>
    <div class="particle p6"></div>
    <div class="particle p7"></div>
    <div class="particle p8"></div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">🤖 Model Overview</div>
        <div class="sidebar-text">
            This app classifies an uploaded face image as <b>Male</b> or <b>Female</b> using a trained machine learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-title">⚙️ Model Details</div>
        <div class="sidebar-text">
            • Input size: <b>{IMG_SIZE} × {IMG_SIZE}</b><br>
            • Preprocessing: Resize + Flatten<br>
            • Output: Male / Female prediction + probabilities
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">📌 How it works</div>
        <div class="sidebar-text">
            1. Upload an image<br>
            2. The app preprocesses it<br>
            3. The model predicts the class<br>
            4. Confidence scores are displayed
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Hero
# -------------------------
st.markdown("""
<div class="hero">
    <h1>✨ AI Gender Classifier Dashboard</h1>
    <p>Upload an image and get a polished AI-powered gender prediction with premium visuals and confidence analytics.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Upload
# -------------------------
st.markdown("## 📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the image. Please upload a valid JPG/PNG file.")
        st.stop()

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    resized = resized.flatten()

    with st.spinner("Analyzing image with AI model..."):
        time.sleep(1)

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    male_prob = float(probability[0]) * 100
    female_prob = float(probability[1]) * 100

    if prediction == 0:
        predicted_label = "Male"
        emoji = "🧑"
    else:
        predicted_label = "Female"
        emoji = "👩"

    # -------------------------
    # Flying hearts celebration effect
    # -------------------------
    st.markdown("""
    <div class="hearts-container">
        <div class="heart h1">💖</div>
        <div class="heart h2">💜</div>
        <div class="heart h3">💗</div>
        <div class="heart h4">💕</div>
        <div class="heart h5">💖</div>
        <div class="heart h6">💜</div>
        <div class="heart h7">💗</div>
        <div class="heart h8">💕</div>
        <div class="heart h9">💖</div>
        <div class="heart h10">💜</div>
        <div class="heart h11">💗</div>
        <div class="heart h12">💕</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Top Row
    # -------------------------
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🖼 Uploaded Image")
        st.markdown('<div class="image-frame"><div class="image-inner">', unsafe_allow_html=True)
        st.image(image, channels="BGR", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-text">{emoji} {predicted_label}</div>
            <div class="prediction-sub">Prediction completed successfully</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Confidence Scores")

        st.markdown(f"""
        <div class="prob-box">
            <div class="metric-title">Male Probability</div>
            <div class="metric-value">{male_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(probability[0]))

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-box">
            <div class="metric-title">Female Probability</div>
            <div class="metric-value">{female_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(probability[1]))
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # Confidence Ring + AI Summary
    # -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        dominant_conf = male_prob if predicted_label == "Male" else female_prob
        ring_html = confidence_ring(
            f"{predicted_label} Confidence",
            dominant_conf,
            "#7c3aed",
            "#06b6d4"
        )
        st.markdown(f'<div class="glass-card">{ring_html}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧠 AI Analysis Summary")

        if dominant_conf >= 85:
            verdict = "High Confidence Prediction"
        elif dominant_conf >= 65:
            verdict = "Moderate Confidence Prediction"
        else:
            verdict = "Low Confidence Prediction"

        st.markdown(f'<div class="badge">{verdict}</div>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Male Probability:** {male_prob:.2f}%")
        st.write(f"**Female Probability:** {female_prob:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # Summary Cards
    # -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class="summary-card">
            <h3>🎯 Prediction</h3>
            <p style="font-size:1.2rem;font-weight:700;">{predicted_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="summary-card">
            <h3>👨 Male Score</h3>
            <p style="font-size:1.2rem;font-weight:700;">{male_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="summary-card">
            <h3>👩 Female Score</h3>
            <p style="font-size:1.2rem;font-weight:700;">{female_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <h2>📁 Upload an image to begin</h2>
        <p style="color:#dbeafe; font-size:1rem;">
            Your AI dashboard will show the prediction, confidence scores, analysis summary, and premium visuals here.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, OpenCV, NumPy and Scikit-learn
</div>
""", unsafe_allow_html=True)

