
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
# CSS Styling + Effects
# -------------------------
st.markdown("""
<style>
/* ===== App Background ===== */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #312e81, #0f766e);
    background-size: 400% 400%;
    animation: gradientBG 14s ease infinite;
    color: white;
    overflow-x: hidden;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ===== Floating Glow Blobs ===== */
.glow {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    z-index: -1;
    opacity: 0.45;
}
.glow1 {
    width: 260px;
    height: 260px;
    background: #7c3aed;
    top: 80px;
    left: 50px;
    animation: float1 10s ease-in-out infinite;
}
.glow2 {
    width: 240px;
    height: 240px;
    background: #06b6d4;
    bottom: 70px;
    right: 80px;
    animation: float2 12s ease-in-out infinite;
}
.glow3 {
    width: 200px;
    height: 200px;
    background: #22c55e;
    top: 45%;
    left: 45%;
    animation: float3 11s ease-in-out infinite;
}

@keyframes float1 {
    0% {transform: translate(0,0);}
    50% {transform: translate(30px, 40px);}
    100% {transform: translate(0,0);}
}
@keyframes float2 {
    0% {transform: translate(0,0);}
    50% {transform: translate(-25px, -35px);}
    100% {transform: translate(0,0);}
}
@keyframes float3 {
    0% {transform: translate(0,0);}
    50% {transform: translate(20px, -25px);}
    100% {transform: translate(0,0);}
}

/* ===== Hero Section ===== */
.hero {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(16px);
    border-radius: 26px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
    animation: fadeUp 1s ease;
    position: relative;
    overflow: hidden;
}

/* moving shine across hero */
.hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: -120%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.20), transparent);
    transform: skewX(-20deg);
    animation: heroShine 4s linear infinite;
}

@keyframes heroShine {
    0% { left: -120%; }
    100% { left: 150%; }
}

.hero h1 {
    font-size: 2.8rem;
    margin-bottom: 10px;
    color: white;
    animation: pulseText 2.5s ease-in-out infinite;
}
.hero p {
    font-size: 1.08rem;
    color: #e2e8f0;
}

@keyframes pulseText {
    0% { text-shadow: 0 0 0px rgba(255,255,255,0.2); }
    50% { text-shadow: 0 0 16px rgba(255,255,255,0.35); }
    100% { text-shadow: 0 0 0px rgba(255,255,255,0.2); }
}

/* ===== Glass Card ===== */
.glass-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.22);
    transition: all 0.35s ease;
    animation: fadeUp 0.9s ease;
}

.glass-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 14px 36px rgba(0,0,0,0.32);
}

/* ===== Image Card ===== */
.image-frame {
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 0 0 rgba(34,197,94,0.0);
    transition: all 0.4s ease;
}
.image-frame:hover {
    transform: scale(1.02);
    box-shadow: 0 0 30px rgba(34,197,94,0.35);
}

/* ===== Prediction Card ===== */
.prediction-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.88), rgba(16,185,129,0.78));
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 30px rgba(16,185,129,0.35);
    animation: pulseGlow 2.2s infinite;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

/* shimmer inside prediction card */
.prediction-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: -130%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent);
    transform: skewX(-20deg);
    animation: shimmer 3.2s linear infinite;
}

@keyframes shimmer {
    0% { left: -130%; }
    100% { left: 150%; }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 rgba(16,185,129,0.3); }
    50% { box-shadow: 0 0 30px rgba(16,185,129,0.55); }
    100% { box-shadow: 0 0 0 rgba(16,185,129,0.3); }
}

.prediction-text {
    font-size: 2rem;
    font-weight: 800;
    animation: floatLabel 2.5s ease-in-out infinite;
}
.prediction-sub {
    font-size: 1rem;
    color: #ecfdf5;
    margin-top: 8px;
}
@keyframes floatLabel {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
    100% { transform: translateY(0px); }
}

/* ===== Probability Boxes ===== */
.prob-box {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 16px;
    border-radius: 16px;
    margin-bottom: 12px;
    transition: 0.3s ease;
}
.prob-box:hover {
    transform: scale(1.02);
    box-shadow: 0 0 22px rgba(255,255,255,0.08);
}
.metric-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f8fafc;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
}

/* ===== Upload Box ===== */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 2px dashed rgba(255,255,255,0.28);
    border-radius: 18px;
    padding: 12px;
    box-shadow: inset 0 0 20px rgba(255,255,255,0.04);
    transition: 0.3s ease;
}
section[data-testid="stFileUploader"]:hover {
    border-color: #60a5fa;
    box-shadow: 0 0 22px rgba(96,165,250,0.20);
}

/* ===== Better Progress Bar ===== */
div[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #22c55e, #06b6d4, #7c3aed) !important;
    border-radius: 999px !important;
}

/* ===== Summary Cards ===== */
.summary-card {
    text-align: center;
    padding: 18px;
    border-radius: 18px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.20);
    transition: all 0.3s ease;
    animation: fadeUp 1s ease;
}
.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.28);
}

/* ===== Fade Animation ===== */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== Footer ===== */
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 25px;
    font-size: 0.95rem;
}
</style>

<div class="glow glow1"></div>
<div class="glow glow2"></div>
<div class="glow glow3"></div>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="hero">
    <h1>✨ Male vs Female Image Classifier</h1>
    <p>Upload an image and let the model predict the gender with a polished AI dashboard experience.</p>
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
        st.error("Could not read the image. Please upload a valid image file.")
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
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
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

