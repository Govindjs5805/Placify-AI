import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placify AI – Placement Predictor",
    layout="wide"
)

# ─── Global CSS: Dark Blue / Black Theme ──────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Base Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(ellipse at top left, #050d1a 0%, #020812 60%, #000000 100%) !important;
        color: #cdd9f0 !important;
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero Banner ── */
    .hero-banner {
        text-align: center;
        padding: 52px 20px 36px;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #60a5fa;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        padding: 6px 18px;
        border-radius: 50px;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 54px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 14px 0;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }
    .hero-sub {
        color: #64748b;
        font-size: 16px;
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #1d4ed8, #60a5fa, #1d4ed8);
        border-radius: 4px;
        margin: 22px auto 0;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: linear-gradient(145deg, rgba(13, 25, 48, 0.85) 0%, rgba(7, 15, 30, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 20px;
        padding: 32px 28px;
        margin-bottom: 20px;
        box-shadow:
            0 0 0 1px rgba(59, 130, 246, 0.06),
            0 20px 60px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    /* ── Section Headings ── */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.3px;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-sub {
        font-size: 13px;
        color: #475569;
        margin: 0 0 28px 0;
        font-weight: 400;
    }

    /* ── Input Fields ── */
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stTextInput"] label p {
        color: #94a3b8 !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        letter-spacing: 0.6px !important;
        text-transform: uppercase !important;
        margin-bottom: 6px !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: rgba(15, 28, 55, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.22) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        outline: none !important;
    }
    div[data-testid="stNumberInput"] input:hover,
    div[data-testid="stTextInput"] input:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
    }

    /* ── Step arrow buttons on number_input ── */
    div[data-testid="stNumberInput"] button {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: rgba(59, 130, 246, 0.25) !important;
        color: #93c5fd !important;
    }

    /* ── Predict Button ── */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4), 0 0 0 1px rgba(59,130,246,0.2) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.55), 0 0 0 1px rgba(59,130,246,0.35) !important;
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%) !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0) !important;
    }

    /* ── Package Hero Metric ── */
    .package-hero {
        background: linear-gradient(145deg, rgba(29, 78, 216, 0.15) 0%, rgba(37, 99, 235, 0.08) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .package-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(59,130,246,0.06) 0%, transparent 65%);
        pointer-events: none;
    }
    .package-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #60a5fa;
        margin: 0 0 10px 0;
    }
    .package-value {
        font-size: 1000px;
        font-weight: 800;
        background: linear-gradient(135deg, #93c5fd, #ffffff, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
        letter-spacing: -2px;
    }
    .package-unit {
        font-size: 18px;
        font-weight: 500;
        color: #60a5fa;
        margin: 8px 0 0 0;
        -webkit-text-fill-color: #60a5fa;
    }

    /* ── Readiness Score Card ── */
    .readiness-card {
        background: rgba(10, 20, 40, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* ── Progress Bar ── */
    .progress-wrap {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 100px;
        height: 8px;
        margin-top: 10px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 0.5s ease;
    }

    /* ── Strategy Tip Cards ── */
    .tip-card {
        background: rgba(10, 20, 42, 0.75);
        border: 1px solid rgba(59, 130, 246, 0.14);
        border-left: 3px solid #2563eb;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 13.5px;
        color: #94a3b8;
        line-height: 1.6;
        transition: border-color 0.2s;
    }
    .tip-card:hover {
        border-left-color: #60a5fa;
        border-color: rgba(96, 165, 250, 0.2);
    }
    .tip-card b {
        color: #bfdbfe;
        font-weight: 600;
    }

    /* ── Input Group Divider ── */
    .input-divider {
        border: none;
        border-top: 1px solid rgba(59, 130, 246, 0.1);
        margin: 20px 0;
    }

    /* ── Column gaps fix ── */
    [data-testid="column"] { padding: 0 10px !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #020812; }
    ::-webkit-scrollbar-thumb { background: #1d4ed8; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #2563eb; }
    </style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    return joblib.load('model/placement_model.pkl')

try:
    model = load_ml_model()
except FileNotFoundError:
    st.error("❌ Model file not found! Run `python train.py` first.")
    st.stop()

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">AI-Powered · Placement Intelligence</div>
        <h1 class="hero-title">Placify AI</h1>
        <p class="hero-sub">Enter your academic profile to predict your placement package &amp; get personalised strategies.</p>
        <div class="hero-divider"></div>
    </div>
""", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.05, 1.3], gap="large")

# ╔══════════════════════════════════════════════════════╗
# ║  LEFT COLUMN – Input Form                           ║
# ╚══════════════════════════════════════════════════════╝
with col1:
    st.markdown("""
        <p class="section-title">Student Performance Metrics</p>
        <p class="section-sub">Fill in your academic & skill details below</p>
    """, unsafe_allow_html=True)

    # ── Primary Metrics ──────────────────────────────────
    cgpa = st.number_input(
        "Cumulative GPA (CGPA)",
        min_value=6.0, max_value=10.0,
        value=8.5, step=0.01,
        format="%.2f",
        help="Your current CGPA on a scale of 6.0 – 10.0"
    )

    dsa_solved = st.number_input(
        "DSA Problems Solved (LeetCode / GFG)",
        min_value=0, max_value=600,
        value=50, step=10,
        help="Total number of DSA problems solved across platforms"
    )

    comm_score = st.number_input(
        "Communication Skills Score (1 – 10)",
        min_value=1, max_value=10,
        value=7, step=1,
        help="Self-assessed communication & soft-skill score"
    )

    st.markdown('<hr class="input-divider">', unsafe_allow_html=True)

    # ── Secondary Metrics (3-col grid) ───────────────────
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        projects = st.number_input(
            "Projects",
            min_value=0, max_value=10,
            value=2, step=1,
            help="Number of projects completed"
        )
    with sub_col2:
        internships = st.number_input(
            "Internships",
            min_value=0, max_value=5,
            value=1, step=1,
            help="Number of internships completed"
        )
    with sub_col3:
        certs = st.number_input(
            "Certifications",
            min_value=0, max_value=10,
            value=1, step=1,
            help="Number of certifications earned"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button("⚡  Predict My Package", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════╗
# ║  RIGHT COLUMN – Analytics & Output                  ║
# ╚══════════════════════════════════════════════════════╝
with col2:
    st.markdown("""
        <p class="section-title">Analytics &amp; Prediction</p>
        <p class="section-sub">Real-time analysis based on your profile</p>
    """, unsafe_allow_html=True)

    # ── Compute predictions ───────────────────────────────
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
        (min(projects, 5) / 5 * 15) +
        (min(internships, 3) / 3 * 15) +
        (comm_score / 10 * 10)
    )
    readiness_score = min(100, max(30, readiness_score))

    # ── Package Hero Block ────────────────────────────────
    st.markdown(f"""
        <div class="package-hero">
            <p class="package-label">Estimated Annual Package</p>
            <p class="package-value">₹{predicted_package}</p>
            <p class="package-unit">Lakhs Per Annum</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Readiness Score ───────────────────────────────────
    if readiness_score >= 80:
        band_label = "Excellent Profile "
        band_color = "#34d399"
        bar_gradient = "linear-gradient(90deg, #059669, #34d399)"
    elif readiness_score >= 60:
        band_label = "Steady Profile "
        band_color = "#60a5fa"
        bar_gradient = "linear-gradient(90deg, #1d4ed8, #60a5fa)"
    else:
        band_label = "Needs Improvement "
        band_color = "#f87171"
        bar_gradient = "linear-gradient(90deg, #991b1b, #f87171)"

    st.markdown(f"""
        <div style="background: rgba(10,20,42,0.7); border: 1px solid rgba(59,130,246,0.15);
                    border-radius: 14px; padding: 22px 24px; margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                <div>
                    <p style="color:#64748b; font-size:11px; font-weight:600; letter-spacing:2px;
                               text-transform:uppercase; margin:0 0 6px 0;">Placement Readiness</p>
                    <p style="color:{band_color}; font-size:13px; font-weight:600; margin:0;">
                        {band_label}
                    </p>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:38px; font-weight:800; color:#e2e8f0; letter-spacing:-1px;">
                        {readiness_score}
                    </span>
                    <span style="font-size:16px; color:#475569; font-weight:500;">/100</span>
                </div>
            </div>
            <div class="progress-wrap">
                <div class="progress-fill" style="width:{readiness_score}%; background:{bar_gradient};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Strategy Tips ─────────────────────────────────────
    st.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:14px;'> Growth Strategies</p>", unsafe_allow_html=True)

    tips_shown = 0
    if cgpa < 8.5:
        st.markdown(f'<div class="tip-card"><b>Academic Performance</b> — Your CGPA of {cgpa:.2f} is below 8.5. Improving your GPA unlocks shortlisting at premium-tier companies during on-campus drives.</div>', unsafe_allow_html=True)
        tips_shown += 1
    if dsa_solved < 350:
        st.markdown(f'<div class="tip-card"><b>Coding Assessment</b> — You have solved {dsa_solved} problems. Crossing the 350+ milestone significantly improves your chances of clearing technical screening rounds.</div>', unsafe_allow_html=True)
        tips_shown += 1
    if internships < 2:
        st.markdown(f'<div class="tip-card"><b>Industry Exposure</b> — Add at least one more internship. Recruiters weigh real-world experience heavily when evaluating candidates for higher packages.</div>', unsafe_allow_html=True)
        tips_shown += 1
    if comm_score < 8:
        st.markdown(f'<div class="tip-card"><b>Communication & Soft Skills</b> — A score of {comm_score}/10 leaves room for growth. Practise mock interviews and structured communication to boost offer negotiations.</div>', unsafe_allow_html=True)
        tips_shown += 1
    if certs < 2:
        st.markdown(f'<div class="tip-card"><b>Certifications</b> — Earning recognised certifications (AWS, Google, Microsoft, Coursera) adds credibility and differentiates your profile.</div>', unsafe_allow_html=True)
        tips_shown += 1

    if tips_shown == 0:
        st.markdown('<div class="tip-card" style="border-left-color:#34d399;"><b>Outstanding Profile </b> — You are well-positioned for top-tier placements. Focus on company-specific preparation and networking.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)