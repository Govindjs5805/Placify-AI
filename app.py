import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Placify AI - Dashboard",
    page_icon="🚀",
    layout="wide"
)

# 2. Complete CSS Makeover (Fixed containers, typography, and contrast)
st.markdown("""
    <style>
    /* Main App Background Override */
    .stApp {
        background-color: #0b0f14 !important;
        color: #e6edf3 !important;
    }
    
    /* Clean Cards for Sections */
    .custom-card {
        background-color: #12171e;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
    }
    
    /* Headers Styling */
    .neon-title {
        color: #00ff66 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
    }
    
    /* Metric Box Matrix Look */
    .metric-box {
        background: linear-gradient(135deg, #0a160f 0%, #12171e 100%);
        border: 2px solid #00ff66;
        box-shadow: 0px 0px 20px rgba(0, 255, 102, 0.15);
        border-radius: 12px;
        padding: 30px 20px;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Strategy Improvement Alerts */
    .strategy-box {
        background-color: #161c24;
        border-left: 4px solid #00ff66;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #e6edf3;
        font-size: 14px;
    }

    /* Force Streamlit Sliders to match Dark/Green Theme */
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        background-color: #21262d;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #00ff66 !important;
        box-shadow: 0px 0px 8px #00ff66;
    }
    div[data-testid="stThumbValue"] {
        color: #00ff66 !important;
        font-family: monospace;
        font-size: 16px;
    }
    
    /* Input Labels Readability Fix */
    label[data-testid="stWidgetLabel"] p {
        color: #8b949e !important;
        font-size: 14px !important;
        font-weight: 500;
    }

    /* Premium Action Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff66 0%, #00cc52 100%) !important;
        color: #0b0f14 !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 20px !important;
        box-shadow: 0 4px 15px rgba(0, 255, 102, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.25s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(0, 255, 102, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load ML Engine
@st.cache_resource
def load_ml_model():
    return joblib.load('model/placement_model.pkl')

try:
    model = load_ml_model()
except FileNotFoundError:
    st.error("❌ Model file not found! Run 'python train.py' first.")
    st.stop()

# 4. App Structure layout
st.markdown("<h1 style='text-align: center; color: #ffffff; font-family: monospace; letter-spacing: 2px; margin-bottom: 30px;'>🚀 PLACIFY AI</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1.3], gap="large")

# Left Column: User Inputs Form Container
with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="neon-title">📊 Student Performance Metrics</h2>', unsafe_allow_html=True)
    
    cgpa = st.slider("Cumulative GPA (CGPA)", min_value=6.0, max_value=10.0, value=9.57, step=0.01)
    dsa_solved = st.slider("DSA Problems Solved (LeetCode/GFG)", min_value=0, max_value=600, value=50, step=10)
    
    # Grid split for specific parameters
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        projects = st.number_input("Projects Completed", min_value=1, max_value=5, value=2)
    with sub_col2:
        internships = st.number_input("Internships Done", min_value=0, max_value=3, value=1)
    with sub_col3:
        certs = st.number_input("Certifications", min_value=0, max_value=5, value=1)
        
    comm_score = st.slider("Communication Skills Scale", min_value=1, max_value=10, value=7)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Calculate Target Valuation ⚡", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Right Column: Analytics & Recommendation Engines
with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="neon-title">🎯 Analytics & Predictions</h2>', unsafe_allow_html=True)
    
    # Calculate values instantly for reactive data flow mapping
    input_data = pd.DataFrame([{
        'CGPA': cgpa,
        'Projects': projects,
        'Internships': internships,
        'DSA_Solved': dsa_solved,
        'Certifications': certs,
        'Communication_Score': comm_score
    }])
    
    raw_prediction = model.predict(input_data)[0]
    predicted_package = max(3.0, round(raw_prediction, 1))
    
    readiness_score = int(
        ((cgpa - 6) / 4 * 35) + 
        (dsa_solved / 600 * 25) + 
        (projects / 5 * 15) + 
        (internships / 3 * 15) + 
        (comm_score / 10 * 10)
    )
    readiness_score = min(100, max(30, readiness_score))

    # HTML Blocks rendering with corrected classes 
    st.markdown(f"""
        <div class="metric-box">
            <p style="font-size: 16px; margin: 0 0 8px 0; color: #8b949e; uppercase; letter-spacing: 1px;">Estimated Package Value</p>
            <h1 style="font-size: 50px; margin: 0; color: #00ff66; font-weight: bold;">₹{predicted_package} LPA</h1>
        </div>
    """, unsafe_allow_html=True)
    
    band_label = "Excellent Profile 🔥" if readiness_score >= 80 else ("Steady Profile 📈" if readiness_score >= 60 else "Needs Improvement ⚠️")
    band_color = "#00ff66" if readiness_score >= 60 else "#ff3333"
    
    st.markdown(f"""
        <div style="background-color: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #21262d; margin-bottom: 25px;">
            <p style="color: #8b949e; margin: 0 0 5px 0; font-size: 14px;">Placement Readiness Band</p>
            <h2 style="margin: 0; font-size: 34px; color: #ffffff; font-weight: bold;">{readiness_score} <span style="font-size: 18px; color: #8b949e; font-weight: normal;">/ 100</span></h2>
            <p style="color: {band_color}; font-weight: 700; margin: 8px 0 0 0; font-size: 13px; letter-spacing: 0.5px;">{band_label}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 18px; color: #ffffff; margin-bottom: 15px;'>💡 Tailored Development Strategies:</h3>", unsafe_allow_html=True)
    
    # Conditional notification engine output blocks
    if cgpa < 8.5:
        st.markdown(f'<div class="strategy-box"><b>Academic Metric</b>: Your CGPA sits below 8.5. Focusing on GPA metrics unlocks initial premium tier target screenings.</div>', unsafe_allow_html=True)
    if dsa_solved < 350:
        st.markdown(f'<div class="strategy-box"><b>Coding Assessment</b>: Currently at {dsa_solved} solved problems. Push to cross the 350+ milestone to clear technical rounds smoothly.</div>', unsafe_allow_html=True)
    if internships < 2:
        st.markdown(f'<div class="strategy-box"><b>Work Exposure</b>: Adding another corporate internship expands evaluation package metrics significantly.</div>', unsafe_allow_html=True)
    if comm_score < 8:
        st.markdown(f'<div class="strategy-box"><b>Interview Presentation</b>: Practicing presentation clarity creates tier value negotiation leverage.</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)