
import streamlit as st
import numpy as np
import cv2
import joblib
import time

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="✨",
    layout="wide"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("modell.pkl")
IMG_SIZE = 64

# -------------------------
# Premium CSS Styling + Beautiful Effects
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
        radial-gradient(circle at 50% 85%, rgba(34, 197, 94, 0.20), transparent 30%),
        linear-gradient(135deg, #081120 0%, #0f172a 45%, #111827 100%);
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
    filter: blur(50px);
    animation: auroraMove 14s ease-in-out infinite alternate;
    z-index: -3;
}

@keyframes auroraMove {
    0%   { transform: translate3d(0, 0, 0) scale(1); }
    50%  { transform: translate3d(25px, -20px, 0) scale(1.04); }
    100% { transform: translate3d(-20px, 18px, 0) scale(1.02); }
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
    width: 10px;
    height: 10px;
    background: rgba(255,255,255,0.12);
    border-radius: 50%;
    box-shadow: 0 0 18px rgba(255,255,255,0.18);
    animation: drift linear infinite;
}
.p1 { top: 12%; left: 14%; animation-duration: 13s; width: 8px; height: 8px; }
.p2 { top: 25%; left: 80%; animation-duration: 16s; width: 12px; height: 12px; }
.p3 { top: 68%; left: 20%; animation-duration: 18s; width: 9px; height: 9px; }
.p4 { top: 76%; left: 72%; animation-duration: 14s; width: 7px; height: 7px; }
.p5 { top: 45%; left: 50%; animation-duration: 20s; width: 10px; height: 10px; }
.p6 { top: 15%; left: 55%; animation-duration: 17s; width: 6px; height: 6px; }
.p7 { top: 58%; left: 88%; animation-duration: 15s; width: 11px; height: 11px; }
.p8 { top: 84%; left: 42%; animation-duration: 19s; width: 8px; height: 8px; }

@keyframes drift {
    0%   { transform: translateY(0px) translateX(0px); opacity: 0.2; }
    25%  { opacity: 0.55; }
    50%  { transform: translateY(-28px) translateX(18px); opacity: 0.3; }
    75%  { opacity: 0.6; }
    100% { transform: translateY(0px) translateX(0px); opacity: 0.2; }
}

/* =========================
   HERO SECTION
========================= */
.hero {
    position: relative;
    overflow: hidden;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(16px);
    border-radius: 28px;
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
    font-size: 2.8rem;
    color: #ffffff;
    text-shadow: 0 0 18px rgba(255,255,255,0.10);
}
.hero p {
    margin-top: 10px;
    color: #dbeafe;
    font-size: 1.05rem;
}

/* =========================
   GLASS CARD
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
    letter-spacing: 0.3px;
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
   PROGRESS BAR
========================= */
div[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e) !important;
    border-radius: 999px !important;
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
   ANIMATIONS + FOOTER
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
# HERO
# -------------------------
st.markdown("""
<div class="hero">
    <h1>✨ Male vs Female Image Classifier</h1>
    <p>Upload an image and get a polished AI-powered prediction with elegant visuals and confidence scores.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# UPLOAD
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

    with st.spinner("Analyzing image..."):
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

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="summary-card">
            <h3>🎯 Prediction</h3>
            <p style="font-size:1.2rem;font-weight:700;">{predicted_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="summary-card">
            <h3>👨 Male Score</h3>
            <p style="font-size:1.2rem;font-weight:700;">{male_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="summary-card">
            <h3>👩 Female Score</h3>
            <p style="font-size:1.2rem;font-weight:700;">{female_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <h3>📁 No image uploaded yet</h3>
        <p>Please upload a JPG, JPEG, or PNG image to begin classification.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, OpenCV, NumPy and Scikit-learn
</div>
""", unsafe_allow_html=True)

