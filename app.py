
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
# Custom CSS for Styling + Motion
# -------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #312e81, #0f766e);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

/* Animated background */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating glowing circles */
.background-blur {
    position: fixed;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    filter: blur(80px);
    z-index: -1;
    opacity: 0.55;
}

.blur1 {
    top: 80px;
    left: 50px;
    background: #7c3aed;
    animation: float1 8s ease-in-out infinite;
}

.blur2 {
    bottom: 80px;
    right: 80px;
    background: #06b6d4;
    animation: float2 10s ease-in-out infinite;
}

@keyframes float1 {
    0% {transform: translateY(0px) translateX(0px);}
    50% {transform: translateY(40px) translateX(20px);}
    100% {transform: translateY(0px) translateX(0px);}
}

@keyframes float2 {
    0% {transform: translateY(0px) translateX(0px);}
    50% {transform: translateY(-30px) translateX(-20px);}
    100% {transform: translateY(0px) translateX(0px);}
}

/* Hero section */
.hero {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
    animation: fadeUp 1s ease;
}

.hero h1 {
    font-size: 2.7rem;
    margin-bottom: 10px;
    color: white;
}

.hero p {
    font-size: 1.1rem;
    color: #e2e8f0;
}

/* Glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.22);
    transition: all 0.3s ease;
    animation: fadeUp 0.8s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 36px rgba(0,0,0,0.30);
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.85), rgba(16,185,129,0.75));
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 30px rgba(16,185,129,0.35);
    animation: pulseGlow 2s infinite;
    margin-bottom: 18px;
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 rgba(16,185,129,0.3); }
    50% { box-shadow: 0 0 28px rgba(16,185,129,0.55); }
    100% { box-shadow: 0 0 0 rgba(16,185,129,0.3); }
}

.prediction-text {
    font-size: 2rem;
    font-weight: 800;
}

.prediction-sub {
    font-size: 1rem;
    color: #ecfdf5;
    margin-top: 8px;
}

/* Probability boxes */
.prob-box {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 16px;
    border-radius: 16px;
    margin-bottom: 14px;
    transition: 0.3s ease;
}

.prob-box:hover {
    transform: scale(1.02);
}

.metric-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f8fafc;
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff;
}

/* Upload area styling */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 2px dashed rgba(255,255,255,0.25);
    border-radius: 18px;
    padding: 10px;
}

/* Fade animation */
@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 25px;
    font-size: 0.95rem;
}
</style>

<div class="background-blur blur1"></div>
<div class="background-blur blur2"></div>
""", unsafe_allow_html=True)

# -------------------------
# Hero Header
# -------------------------
st.markdown("""
<div class="hero">
    <h1>✨ Male vs Female Image Classifier</h1>
    <p>Upload an image and let the AI model predict whether the person is male or female with confidence scores.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Upload Section
# -------------------------
st.markdown("## 📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# -------------------------
# If file uploaded
# -------------------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the image. Please upload a valid JPG/PNG file.")
        st.stop()

    # Preprocess
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    resized = resized.flatten()

    # Prediction animation feel
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

    # Layout
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🖼 Uploaded Image")
        st.image(image, channels="BGR", use_container_width=True)
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
        st.progress(probability[0])

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-box">
            <div class="metric-title">Female Probability</div>
            <div class="metric-value">{female_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(probability[1])

        st.markdown('</div>', unsafe_allow_html=True)

    # Summary row
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <h3>🎯 Prediction</h3>
            <p style="font-size:1.2rem;font-weight:700;">{predicted_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <h3>👨 Male Score</h3>
            <p style="font-size:1.2rem;font-weight:700;">{male_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
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

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, OpenCV, NumPy and Scikit-learn
</div>
""", unsafe_allow_html=True)

