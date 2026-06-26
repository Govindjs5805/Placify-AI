import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placify AI – Placement Predictor",
    page_icon="🎯",
    layout="wide"
)
# ─── Dark / Light Mode State ──────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─── Theme Tokens ─────────────────────────────────────────────────────────────
if st.session_state.dark_mode:
    T = {
        "bg":        "#090909",
        "bg2":       "#111111",
        "card":      "rgba(16,16,16,0.97)",
        "inp":       "rgba(18,18,18,0.92)",
        "b1":        "rgba(255,255,255,0.07)",
        "b2":        "rgba(255,255,255,0.12)",
        "t1":        "#f2f2f2",
        "t2":        "#9a9a9a",
        "t3":        "#555555",
        "g4":        "#4ade80",
        "g5":        "#22c55e",
        "g6":        "#16a34a",
        "gg":        "rgba(34,197,94,0.16)",
        "ggs":       "rgba(34,197,94,0.28)",
        "gr8":       "#1a1a1a",
        "gr7":       "#2a2a2a",
        "gr6":       "#3a3a3a",
        "bggreen":   "rgba(34,197,94,0.28)",
        # chart colors
        "chbg":      "#0d0d0d",
        "chcard":    "#141414",
        "chtext":    "#c0c0c0",
        "chtext2":   "#666666",
        "chgray":    "#3a3a3a",
        # pkg box
        "pkgbg":     "linear-gradient(135deg,#0d2818 0%,#0f3320 30%,#0a2214 60%,#061409 100%)",
        "pkgnumcol": "linear-gradient(135deg,#86efac 0%,#4ade80 40%,#ffffff 80%)",
        "pkgunit":   "#86efac",
        "pkgborder": "rgba(34,197,94,0.38)",
        # input card gradient  
        "inpcard":   "linear-gradient(145deg,rgba(13,30,15,0.95) 0%,rgba(10,20,10,0.98) 100%)",
        "inpcborder":"rgba(34,197,94,0.22)",
        # metric mini cards
        "mc1bg":     "linear-gradient(135deg,rgba(34,197,94,0.08) 0%,rgba(22,163,74,0.04) 100%)",
        "mc2bg":     "linear-gradient(135deg,rgba(34,197,94,0.08) 0%,rgba(22,163,74,0.04) 100%)",
        "mc3bg":     "linear-gradient(135deg,rgba(34,197,94,0.08) 0%,rgba(22,163,74,0.04) 100%)",
        "mc4bg":     "linear-gradient(135deg,rgba(34,197,94,0.08) 0%,rgba(22,163,74,0.04) 100%)",
        "mc1b":      "rgba(34,197,94,0.18)",
        "mc2b":      "rgba(34,197,94,0.18)",
        "mc3b":      "rgba(34,197,94,0.18)",
        "mc4b":      "rgba(34,197,94,0.18)",
        "mc1c":      "#4ade80",
        "mc2c":      "#4ade80",
        "mc3c":      "#4ade80",
        "mc4c":      "#4ade80",
        "togglebg":  "#1a1a1a",
        "toggleb":   "rgba(255,255,255,0.12)",
        "togglelbl": "#f2f2f2",
        "toggleicon":"☀️",
        "toggletxt": "Light Mode",
    }
else:
    T = {
        "bg":        "#f0f4f0",
        "bg2":       "#ffffff",
        "card":      "rgba(255,255,255,0.97)",
        "inp":       "rgba(243,246,243,0.95)",
        "b1":        "rgba(0,0,0,0.07)",
        "b2":        "rgba(0,0,0,0.11)",
        "t1":        "#111111",
        "t2":        "#444444",
        "t3":        "#888888",
        "g4":        "#16a34a",
        "g5":        "#15803d",
        "g6":        "#166534",
        "gg":        "rgba(22,163,74,0.12)",
        "ggs":       "rgba(22,163,74,0.24)",
        "gr8":       "#eeeeee",
        "gr7":       "#dedede",
        "gr6":       "#cccccc",
        "bggreen":   "rgba(22,163,74,0.22)",
        # chart colors
        "chbg":      "#f8faf8",
        "chcard":    "#ffffff",
        "chtext":    "#333333",
        "chtext2":   "#888888",
        "chgray":    "#cccccc",
        # pkg box
        "pkgbg":     "linear-gradient(135deg,#f0fdf4 0%,#dcfce7 40%,#bbf7d0 100%)",
        "pkgnumcol": "linear-gradient(135deg,#15803d 0%,#16a34a 55%,#166534 100%)",
        "pkgunit":   "#15803d",
        "pkgborder": "rgba(22,163,74,0.45)",
        # input card gradient
        "inpcard":   "linear-gradient(145deg,rgba(240,253,244,0.98) 0%,rgba(220,252,231,0.9) 100%)",
        "inpcborder":"rgba(22,163,74,0.30)",
        # metric mini cards
        "mc1bg":     "linear-gradient(135deg,rgba(22,163,74,0.06) 0%,rgba(34,197,94,0.02) 100%)",
        "mc2bg":     "linear-gradient(135deg,rgba(22,163,74,0.06) 0%,rgba(34,197,94,0.02) 100%)",
        "mc3bg":     "linear-gradient(135deg,rgba(22,163,74,0.06) 0%,rgba(34,197,94,0.02) 100%)",
        "mc4bg":     "linear-gradient(135deg,rgba(22,163,74,0.06) 0%,rgba(34,197,94,0.02) 100%)",
        "mc1b":      "rgba(22,163,74,0.18)",
        "mc2b":      "rgba(22,163,74,0.18)",
        "mc3b":      "rgba(22,163,74,0.18)",
        "mc4b":      "rgba(22,163,74,0.18)",
        "mc1c":      "#15803d",
        "mc2c":      "#15803d",
        "mc3c":      "#15803d",
        "mc4c":      "#15803d",
        "togglebg":  "#e8f5e9",
        "toggleb":   "rgba(22,163,74,0.25)",
        "togglelbl": "#111111",
        "toggleicon":"🌙",
        "toggletxt": "Dark Mode",
    }

# ─── Inject dynamic CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html,body,[class*="css"]{{ font-family:'Inter',sans-serif!important; }}
.stApp{{ background:{T['bg']}!important; color:{T['t1']}!important; min-height:100vh; }}
#MainMenu,footer,header{{ visibility:hidden; }}
.block-container{{ padding-top:0!important; padding-bottom:60px!important; max-width:1320px!important; }}
.stMarkdown p{{ color:{T['t2']}; }}

/* ══ HERO ══ */
.hero{{position:relative;text-align:left;padding:52px 24px 36px;overflow:hidden;}}
.hero::before{{content:'';position:absolute;top:-110px;left:20%;transform:translateX(-50%);
    width:700px;height:420px;
    background:radial-gradient(ellipse,rgba(34,197,94,0.10) 0%,transparent 68%);
    pointer-events:none;z-index:0;}}
.eyebrow{{position:relative;display:inline-flex;align-items:center;gap:8px;
    background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.22);
    color:{T['g4']};font-size:11px;font-weight:600;letter-spacing:3px;
    text-transform:uppercase;padding:6px 18px;border-radius:50px;margin-bottom:20px;
    font-family:'Space Grotesk',sans-serif;}}
.dot{{width:6px;height:6px;border-radius:50%;background:{T['g5']};animation:pdot 2s infinite;}}
@keyframes pdot{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:.45;transform:scale(.65);}}}}
.hero h1.htitle{{font-family: 'Space Grotesk', sans-serif;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 6rem !important; /* Forces the massive scale */
    font-weight: 800 !important;
    letter-spacing: -2.5px !important;
    line-height: 1.05 !important; /* Fixed typo from line-weight */
    text-align: left !important; /* Forces left alignment */
    padding-right: 50px;
    background: linear-gradient(90deg, #ffffff 0%, #00ff66 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    padding-top: 20px;
    padding-bottom: 15px;
    margin: 0 !important; /* Removed 'auto' which causes centering */
    display: block !important;}}
.htitle .acc{{background:linear-gradient(135deg,{T['g4']} 0%,{T['g5']} 55%,#86efac 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.hsub{{position:absolute;color:{T['t2']};font-size:16px;font-weight:400;
    margin:0 0 0 40px;max-width:540px;line-height:1.7;}}
.hdiv{{width:44px;height:2px;
    background:linear-gradient(90deg,transparent,{T['g5']},transparent);
    border-radius:4px;margin:24px 0 0 0;}}

/* ══ DARK MODE TOGGLE BUTTON ══ */
.dm-toggle{{
    display:inline-flex;align-items:center;gap:8px;
    background:{T['togglebg']};border:1px solid {T['toggleb']};
    border-radius:50px;padding:8px 18px;cursor:pointer;
    font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
    color:{T['togglelbl']};transition:all .2s;margin-top:16px;}}
.dm-toggle:hover{{opacity:.85;}}

/* ══ TABS ══ */
div[data-testid="stTabs"] [role="tablist"]{{
    background:{T['bg2']};border:1px solid {T['b1']};
    border-radius:28px;padding:6px;gap:4px;
    justify-content:center;display:flex;margin-bottom:32px;}}
div[data-testid="stTabs"] [role="tab"]{{
    font-family:'Space Grotesk',sans-serif!important;
    font-size:14px!important;font-weight:600!important;
    color:{T['t2']}!important;letter-spacing:0.2px!important;
    border-radius:20px!important;padding:10px 28px!important;
    border:none!important;background:transparent!important;
    transition:all .22s ease!important;}}
div[data-testid="stTabs"] [role="tab"]:hover{{color:{T['t1']}!important;}}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{
    background:linear-gradient(135deg,{T['g6']} 0%,{T['g5']} 60%,{T['g4']} 100%)!important;
    color:#020a04!important;
    box-shadow:0 4px 16px rgba(34,197,94,0.30)!important;}}

/* ══ GLASS CARDS ══ */
.card{{background:{T['card']};border:1px solid {T['b1']};border-radius:28px;
    padding:32px 28px;margin-bottom:20px;
    box-shadow:0 2px 4px rgba(0,0,0,.12),0 16px 48px rgba(0,0,0,.10);
    transition:border-color .25s,box-shadow .25s;}}
.card:hover{{border-color:{T['bggreen']};}}

/* ══ INPUT PANEL HEADER (left panel Predict) ══ */
.inp-panel{{
    margin-bottom:24px;}}

/* ══ SECTION LABELS ══ */
.slabel{{font-size:10.5px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
    color:{T['g5']};margin:0 0 5px 0;font-family:'Space Grotesk',sans-serif;}}
.stitle{{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;
    color:{T['t1']};letter-spacing:-.5px;margin:0 0 4px 0;}}
.ssub{{font-size:13px;color:{T['t3']};margin:0 0 24px 0;line-height:1.55;}}

/* ══ INPUT FIELDS ══ */
div[data-testid="stNumberInput"] label p,
div[data-testid="stTextInput"] label p{{
    color:{T['t2']}!important;font-size:12px!important;font-weight:600!important;
    letter-spacing:.9px!important;text-transform:uppercase!important;
    margin-bottom:6px!important;font-family:'Space Grotesk',sans-serif!important;}}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input{{
    background:{T['inp']}!important;border:1px solid {T['b2']}!important;
    border-radius:14px!important;color:{T['t1']}!important;
    font-size:15px!important;font-weight:500!important;
    padding:11px 14px!important;transition:all .2s ease!important;}}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus{{
    border-color:{T['g5']}!important;
    box-shadow:0 0 0 3px {T['gg']}!important;outline:none!important;}}
div[data-testid="stNumberInput"] input:hover,
div[data-testid="stTextInput"] input:hover{{border-color:rgba(34,197,94,0.40)!important;}}
div[data-testid="stNumberInput"] button{{
    background:{T['gr8']}!important;border:1px solid {T['b2']}!important;
    color:{T['g4']}!important;border-radius:6px!important;transition:all .2s!important;}}
div[data-testid="stNumberInput"] button:hover{{
    background:{T['gg']}!important;border-color:{T['g5']}!important;}}

/* ══ PREDICT BUTTON ══ */
div.stButton > button:first-child{{
    background:linear-gradient(135deg,{T['g6']} 0%,{T['g5']} 60%,{T['g4']} 100%)!important;
    color:#020a04!important;font-weight:700!important;font-size:15px!important;
    font-family:'Space Grotesk',sans-serif!important;border:none!important;
    border-radius:14px!important;padding:14px 28px!important;
    width:100%!important;letter-spacing:.3px!important;
    box-shadow:0 0 0 1px rgba(34,197,94,0.22),0 4px 20px rgba(34,197,94,0.25)!important;
    transition:all .25s cubic-bezier(.4,0,.2,1)!important;cursor:pointer!important;}}
div.stButton > button:first-child:hover{{
    transform:translateY(-2px)!important;
    box-shadow:0 0 0 1px rgba(34,197,94,0.38),0 8px 32px rgba(34,197,94,0.42)!important;}}
div.stButton > button:first-child:active{{transform:translateY(0)!important;}}

/* ══ PACKAGE HERO BOX ══ */
.pkg-box{{
    position:relative;overflow:hidden;border-radius:28px;
    padding:48px 32px 40px;text-align:center;margin-bottom:20px;
    background:{T['pkgbg']};
    border:1px solid {T['pkgborder']};
    box-shadow:0 0 0 1px rgba(34,197,94,0.07),
               0 0 90px rgba(34,197,94,0.12),
               0 24px 64px rgba(0,0,0,0.35),
               inset 0 1px 0 rgba(255,255,255,0.06);}}
.pkg-box::before{{content:'';position:absolute;top:-80px;left:50%;
    transform:translateX(-50%);width:500px;height:320px;
    background:radial-gradient(circle,rgba(34,197,94,0.13) 0%,transparent 65%);pointer-events:none;}}
.pkg-box::after{{content:'';position:absolute;bottom:-50px;right:-50px;
    width:240px;height:240px;border-radius:50%;
    background:radial-gradient(circle,rgba(74,222,128,0.06) 0%,transparent 65%);pointer-events:none;}}
.pkg-eyebrow{{position:relative;display:inline-flex;align-items:center;gap:6px;
    font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
    color:{T['g4']};margin:0 0 20px 0;font-family:'Space Grotesk',sans-serif;}}
.pkg-eyebrow .dot2{{width:5px;height:5px;border-radius:50%;background:{T['g5']};animation:pdot 2s infinite;}}
.pkg-num{{
    position:relative;
    display:flex;
    justify-content:center;
    align-items:baseline;
    margin:10px 0;
}}
.pkg-currency{{
    font-family:'Space Grotesk',sans-serif;
    font-size:42px!important;
    font-weight:700!important;
    color:{T['g4']}!important;
    margin-right:6px;
}}
.pkg-val-large{{
    font-family:'Space Grotesk',sans-serif;
    font-size:96px!important;
    font-weight:800!important;
    letter-spacing:-3px!important;
    line-height:1!important;
    background:{T['pkgnumcol']};
    -webkit-background-clip:text!important;
    -webkit-text-fill-color:transparent!important;
    background-clip:text!important;
    display:inline-block;
}}
.pkg-sep{{position:relative;margin:22px auto 0;width:72%;height:1px;
    background:linear-gradient(90deg,transparent,{T['pkgborder']},transparent);}}
.pkg-unit{{position:relative;font-size:12px;font-weight:700;
    color:{T['pkgunit']};margin:16px 0 0 0;letter-spacing:2px;text-transform:uppercase;
    font-family:'Space Grotesk',sans-serif;}}
.pkg-badge{{position:relative;display:inline-block;margin-top:14px;
    background:rgba(34,197,94,0.10);border:1px solid rgba(34,197,94,0.25);
    border-radius:50px;padding:5px 16px;
    font-size:12px;font-weight:600;letter-spacing:0.5px;
    font-family:'Space Grotesk',sans-serif;}}

/* ══ COLOURED MINI METRIC CARDS (Predict) ══ */
.pmc{{border-radius:20px;padding:20px 16px;text-align:center;margin-bottom:0;}}
.pmc-1{{background:{T['mc1bg']};border:1px solid {T['mc1b']};}}
.pmc-2{{background:{T['mc2bg']};border:1px solid {T['mc2b']};}}
.pmc-3{{background:{T['mc3bg']};border:1px solid {T['mc3b']};}}
.pmc-4{{background:{T['mc4bg']};border:1px solid {T['mc4b']};}}
.pmc-icon{{font-size:24px;margin-bottom:8px;}}
.pmc-val{{font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;
    line-height:1;margin:0 0 4px 0;}}
.pmc-lbl{{font-size:10px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;
    color:{T['t3']};}}

/* ══ PROGRESS BARS ══ */
.pbar-row{{margin-bottom:16px;}}
.pbar-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}}
.pbar-name{{font-size:12.5px;font-weight:500;color:{T['t2']};}}
.pbar-val{{font-size:12px;font-weight:700;font-family:'Space Grotesk',sans-serif;}}
.pbar-track{{background:{T['gr8']};border-radius:100px;height:8px;overflow:hidden;}}
.pbar-fill{{height:100%;border-radius:100px;transition:width .6s cubic-bezier(.4,0,.2,1);}}

/* ══ READINESS BLOCK ══ */
.rblock{{background:{T['card']};border:1px solid {T['b1']};
    border-radius:20px;padding:24px;margin-bottom:20px;}}
.rhead{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;}}
.rbig{{font-family:'Space Grotesk',sans-serif;font-size:44px;font-weight:700;
    color:{T['t1']};line-height:1;letter-spacing:-1.5px;}}
.rdenom{{font-size:18px;color:{T['t3']};font-weight:500;}}
.rlbl{{font-size:10.5px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
    color:{T['t3']};margin:0 0 5px 0;font-family:'Space Grotesk',sans-serif;}}
.rband{{font-size:14px;font-weight:600;margin:0;}}
.ptrack{{background:{T['gr8']};border-radius:100px;height:7px;margin-top:4px;overflow:hidden;}}
.pbar{{height:100%;border-radius:100px;transition:width .6s cubic-bezier(.4,0,.2,1);}}

/* ══ TIP CARDS ══ */
.tips-hdr{{font-size:10.5px;font-weight:700;letter-spacing:2.5px;
    text-transform:uppercase;color:{T['t3']};margin-bottom:12px;
    font-family:'Space Grotesk',sans-serif;}}
.tip{{background:{T['card']};border:1px solid {T['b1']};
    border-left:3px solid {T['gr6']};
    border-radius:0 8px 8px 0;
    padding:14px 16px;margin-bottom:10px;font-size:13.5px;
    color:{T['t2']};line-height:1.6;transition:border-left-color .2s,border-color .2s;}}
.tip:hover{{border-left-color:{T['g5']};border-color:{T['ggs']};}}
.tip b{{color:{T['t1']};font-weight:600;}}
.tip-ok{{border-left-color:{T['g5']}!important;
    background:rgba(34,197,94,0.04)!important;border-color:rgba(34,197,94,0.18)!important;}}

/* ══ INSIGHTS / MODEL METRIC CARDS ══ */
.mcard{{background:{T['card']};border:1px solid {T['b1']};
    border-radius:28px;padding:28px 24px;margin-bottom:16px;
    text-align:center;transition:border-color .25s;}}
.mcard:hover{{border-color:{T['bggreen']};}}
.mcard-icon{{font-size:28px;margin-bottom:10px;}}
.mcard-lbl{{font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
    color:{T['t3']};margin:0 0 8px 0;font-family:'Space Grotesk',sans-serif;}}
.mcard-val{{font-family:'Space Grotesk',sans-serif;font-size:36px;font-weight:700;
    color:{T['g4']};line-height:1;letter-spacing:-1px;margin:0;}}
.mcard-sub{{font-size:12px;color:{T['t3']};margin:6px 0 0 0;line-height:1.4;}}

.coef-row{{display:flex;align-items:center;gap:14px;padding:12px 0;border-bottom:1px solid {T['b1']};}}
.coef-row:last-child{{border-bottom:none;}}
.coef-name{{flex:1;font-size:13.5px;font-weight:500;color:{T['t2']};}}
.coef-bar-wrap{{flex:2;background:{T['gr8']};border-radius:100px;height:6px;overflow:hidden;}}
.coef-bar-inner{{height:100%;border-radius:100px;
    background:linear-gradient(90deg,{T['g6']},{T['g4']});}}
.coef-val{{flex:0 0 56px;text-align:right;font-family:'Space Grotesk',sans-serif;
    font-size:13px;font-weight:600;color:{T['g4']};}}

/* ══ MISC ══ */
.idiv{{border:none;border-top:1px solid {T['b1']};margin:20px 0;}}
[data-testid="column"]{{padding:0 8px!important;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{T['gr6']};border-radius:4px;}}
::-webkit-scrollbar-thumb:hover{{background:{T['g6']};}}
div[data-testid="stImage"] img{{border-radius:20px;}}

/* ══ ANIMATIONS ══ */
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
.anim{{animation:fadeUp .45s cubic-bezier(.4,0,.2,1) both;}}
.d1{{animation-delay:.05s;}} .d2{{animation-delay:.10s;}}
.d3{{animation-delay:.15s;}} .d4{{animation-delay:.20s;}}
@keyframes shimmer{{0%{{background-position:-200% center;}}100%{{background-position:200% center;}}}}
</style>
""", unsafe_allow_html=True)

# ─── Data & Model Loaders ─────────────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    return joblib.load('model/placement_model.pkl')

@st.cache_data
def load_dataset():
    return pd.read_csv('dataset/students.csv')

@st.cache_data
def compute_metrics():
    df = pd.read_csv('dataset/students.csv')
    X = df.drop(columns=['Package'])
    y = df['Package']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    return {
        'mae':   mean_absolute_error(y_test, y_pred),
        'mse':   mean_squared_error(y_test, y_pred),
        'rmse':  np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2':    r2_score(y_test, y_pred),
        'train': X_train.shape[0],
        'test':  X_test.shape[0],
        'coefs': dict(zip(X.columns, mdl.coef_)),
        'intercept': mdl.intercept_,
        'y_test': y_test.values,
        'y_pred': y_pred,
        'X_test': X_test,
        'df': df,
        'model': mdl,
    }

try:
    model = load_ml_model()
except FileNotFoundError:
    st.error("❌ Model file not found! Run `python train.py` first.")
    st.stop()

df_full = load_dataset()
metrics = compute_metrics()

# ─── Hero + Dark Mode Toggle ──────────────────────────────────────────────────
st.markdown(f"""
<div class="hero anim">
    <div class="eyebrow"><span class="dot"></span>AI-Powered &nbsp;·&nbsp; Placement Intelligence</div>
    <h1 class="htitle">Placify <span class="acc">AI</span></h1>
    <div class="hdiv"></div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_model = st.tabs([
    "Predict",
    "Insights",
    "Model",
])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 – PREDICT  (fully redesigned)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_predict:
    col1, col2 = st.columns([1.05, 1.35], gap="large")

    # ── LEFT: Input Form (aligned header) ─────────────────────────────────────
    with col1:
        st.markdown(f"""
        <div class="inp-panel anim d1">
            <p class="slabel">Step 1 — Your Profile</p>
            <h2 class="stitle" style="font-size: 26px; font-weight: 700; margin: 0 0 6px 0; color: {T['t1']};">Academic & Skill Details</h2>
            <p class="ssub" style="margin: 0; color: {T['t3']}; font-size: 13.5px;">Adjust the values below — predictions update in real time.</p>
        </div>
        """, unsafe_allow_html=True)

        cgpa = st.number_input(
            "Cumulative GPA (CGPA)", min_value=6.0, max_value=10.0,
            value=8.5, step=0.01, format="%.2f",
            help="Your current CGPA on a scale of 6.0 – 10.0")

        dsa_solved = st.number_input(
            "DSA Problems Solved (LeetCode / GFG)",
            min_value=0, max_value=600, value=50, step=10,
            help="Total DSA problems solved across all platforms")

        comm_score = st.number_input(
            "Communication Skills Score (1 – 10)",
            min_value=1, max_value=10, value=7, step=1,
            help="Self-assessed communication & soft-skill score")

        st.markdown(f'<hr class="idiv">', unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            projects = st.number_input("Projects", min_value=0, max_value=10, value=2, step=1)
        with sc2:
            internships = st.number_input("Internships", min_value=0, max_value=5, value=1, step=1)
        with sc3:
            certs = st.number_input("Certifications", min_value=0, max_value=10, value=1, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict My Package", use_container_width=True)

        # ── Profile Progress Bars ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="slabel" style="margin-bottom:14px;">Profile Breakdown</p>', unsafe_allow_html=True)

        bars = [
            ("CGPA",           (cgpa - 6) / 4 * 100,        T["g5"],      f"{cgpa:.2f} / 10.0"),
            ("DSA Problems",   dsa_solved / 600 * 100,       T["g5"],      f"{dsa_solved} / 600"),
            ("Communication",  comm_score / 10 * 100,        T["g5"],      f"{comm_score} / 10"),
            ("Projects",       min(projects, 5) / 5 * 100,   T["g5"],      f"{projects} / 5+"),
            ("Internships",    min(internships, 3) / 3 * 100,T["g5"],      f"{internships} / 3+"),
            ("Certifications", min(certs, 5) / 5 * 100,      T["g5"],      f"{certs} / 5+"),
        ]
        bars_html = ""
        for name, pct, color, label in bars:
            bars_html += f"""
            <div class="pbar-row">
              <div class="pbar-header">
                <span class="pbar-name">{name}</span>
                <span class="pbar-val" style="color:{color};">{label}</span>
              </div>
              <div class="pbar-track">
                <div class="pbar-fill" style="width:{pct:.1f}%;background:linear-gradient(90deg,{color}aa,{color});"></div>
              </div>
            </div>"""
        st.markdown(f'<div class="anim d2">{bars_html}</div>', unsafe_allow_html=True)

    # ── RIGHT: Results ────────────────────────────────────────────────────────
    with col2:
        inp = pd.DataFrame([{
            'CGPA': cgpa, 'Projects': projects, 'Internships': internships,
            'DSA_Solved': dsa_solved, 'Certifications': certs,
            'Communication_Score': comm_score
        }])
        raw = model.predict(inp)[0]
        pkg = max(3.0, round(raw, 1))

        score = int(
            ((cgpa - 6) / 4 * 35) +
            (dsa_solved / 600 * 25) +
            (min(projects, 5) / 5 * 15) +
            (min(internships, 3) / 3 * 15) +
            (comm_score / 10 * 10))
        score = min(100, max(30, score))

        # tier
        if pkg >= 20:
            tier, tier_color = "🏆 Top-Tier Package", T["g4"]
        elif pkg >= 15:
            tier, tier_color = "🥈 High Package",      T["g4"]
        elif pkg >= 10:
            tier, tier_color = "📈 Mid-Range Package", T["g5"]
        else:
            tier, tier_color = "📌 Entry-Level Package", T["t2"]

        # ── BIG Package Box ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="pkg-box anim d1">
            <div class="pkg-eyebrow"><span class="dot2"></span>Estimated Annual Package</div>
            <div class="pkg-num">
                <span class="pkg-currency">₹</span><span class="pkg-val-large">{pkg}</span>
            </div>
            <div class="pkg-sep"></div>
            <p class="pkg-unit">Lakhs Per Annum</p>
            <div class="pkg-badge" style="color:{tier_color};border-color:{tier_color}44;">{tier}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4 colourful mini metric cards ─────────────────────────────────────
        cgpa_pct  = int((cgpa - 6) / 4 * 100)
        dsa_pct   = int(dsa_solved / 600 * 100)

        m1c, m2c, m3c, m4c = st.columns(4)
        metric_data = [
            (m1c, "pmc-1", T["mc1c"], "📚", f"{cgpa:.1f}",     "CGPA"),
            (m2c, "pmc-2", T["mc2c"], "💻", f"{dsa_solved}",   "DSA"),
            (m3c, "pmc-3", T["mc3c"], "🎤", f"{comm_score}/10","Comm"),
            (m4c, "pmc-4", T["mc4c"], "💼", f"{internships}",  "Intern"),
        ]
        for col, cls, color, icon, val, lbl in metric_data:
            with col:
                st.markdown(f"""
                <div class="pmc {cls} anim d2">
                    <div class="pmc-icon">{icon}</div>
                    <p class="pmc-val" style="color:{color};">{val}</p>
                    <p class="pmc-lbl">{lbl}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br style='display:block;margin:4px 0;'>", unsafe_allow_html=True)

        # ── Readiness Score ───────────────────────────────────────────────────
        if score >= 80:
            band, bc, bg = "🏆 Excellent Profile", T["g4"], f"linear-gradient(90deg,{T['g6']},{T['g4']})"
        elif score >= 60:
            band, bc, bg = "📈 Steady Profile",    T["g5"], f"linear-gradient(90deg,{T['g6']},{T['g5']})"
        else:
            band, bc, bg = "🔧 Needs Improvement", T["t2"], f"linear-gradient(90deg,{T['t3']},{T['t2']})"

        st.markdown(f"""
        <div class="rblock anim d3">
            <div class="rhead">
                <div>
                    <p class="rlbl">Placement Readiness Score</p>
                    <p class="rband" style="color:{bc};">{band}</p>
                </div>
                <div style="text-align:right;">
                    <span class="rbig">{score}</span>
                    <span class="rdenom">/100</span>
                </div>
            </div>
            <div class="ptrack"><div class="pbar" style="width:{score}%;background:{bg};"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mini profile chart (radar as bar) ────────────────────────────────
        DARK_BG  = T["chbg"]
        CARD_BG  = T["chcard"]
        GREEN    = "#22c55e"
        GREEN2   = "#4ade80"
        GRAY     = T["chgray"]
        TEXT     = T["chtext"]
        TEXT2    = T["chtext2"]

        plt.rcParams.update({
            'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
            'axes.edgecolor': GRAY, 'axes.labelcolor': TEXT,
            'xtick.color': TEXT2, 'ytick.color': TEXT2,
            'text.color': TEXT, 'grid.color': GRAY, 'grid.linewidth': 0.4,
        })

        # Profile strength horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        labels  = ["CGPA", "DSA\nSolved", "Comm.\nScore", "Projects", "Internships", "Certif."]
        values  = [
            (cgpa - 6) / 4 * 100,
            dsa_solved / 600 * 100,
            comm_score / 10 * 100,
            min(projects, 5) / 5 * 100,
            min(internships, 3) / 3 * 100,
            min(certs, 5) / 5 * 100,
        ]
        bar_colors = [plt.cm.Greens(0.45 + 0.5 * (val / 100.0)) for val in values]
        bars = ax.barh(labels, values, color=bar_colors, edgecolor='none', height=0.55)
        for bar, val in zip(bars, values):
            ax.text(min(val + 2, 98), bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va='center', ha='left', fontsize=8.5,
                    color=TEXT, fontweight='600')
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score %", fontsize=9, color=TEXT2)
        ax.set_title("Your Profile Strength", fontsize=11, fontweight='bold', color=TEXT, pad=10)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(axis='x', alpha=0.20)
        ax.tick_params(left=False)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Growth Tips ───────────────────────────────────────────────────────
        st.markdown(f"<p class='tips-hdr anim d3' style='margin-top:8px;'>Growth Strategies</p>", unsafe_allow_html=True)
        shown = 0
        if cgpa < 8.5:
            st.markdown(f'<div class="tip anim d3"><b>Academic Performance</b> — CGPA {cgpa:.2f} is below 8.5. Improving your GPA unlocks shortlisting at premium-tier companies.</div>', unsafe_allow_html=True)
            shown += 1
        if dsa_solved < 350:
            st.markdown(f'<div class="tip anim d3"><b>Coding Assessment</b> — {dsa_solved} problems solved. Crossing 350+ significantly boosts your technical screening pass rate.</div>', unsafe_allow_html=True)
            shown += 1
        if internships < 2:
            st.markdown(f'<div class="tip anim d4"><b>Industry Exposure</b> — Add at least one more internship. Recruiters weigh real-world experience heavily for higher packages.</div>', unsafe_allow_html=True)
            shown += 1
        if comm_score < 8:
            st.markdown(f'<div class="tip anim d4"><b>Communication & Soft Skills</b> — Score {comm_score}/10. Practise mock interviews to boost offer negotiations.</div>', unsafe_allow_html=True)
            shown += 1
        if certs < 2:
            st.markdown(f'<div class="tip anim d4"><b>Certifications</b> — Add recognised certs (AWS, Google, Microsoft, Coursera) to stand out.</div>', unsafe_allow_html=True)
            shown += 1
        if shown == 0:
            st.markdown('<div class="tip tip-ok anim d3"><b>Outstanding Profile 🎯</b> — You\'re well-positioned for top-tier placements. Focus on company-specific prep and networking.</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 – INSIGHTS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_insights:
    st.markdown("""
    <div class="anim">
        <p class="slabel">Data Insights</p>
        <p class="stitle">Dataset Exploration & Visualisations</p>
        <p class="ssub">Explore the student placement dataset that powers the prediction model.</p>
    </div>
    """, unsafe_allow_html=True)

    df = df_full.copy()

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("📋", "Total Records",  f"{len(df):,}",                          "student profiles"),
        ("📐", "Features Used",  "6",                                      "input variables"),
        ("💰", "Avg Package",    f"₹{df['Package'].mean():.1f} L",         "mean LPA"),
        ("🎯", "Package Range",  f"₹{df['Package'].min():.0f}–{df['Package'].max():.0f} L", "min to max"),
    ]
    for col, (icon, lbl, val, sub) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f"""
            <div class="mcard anim">
                <div class="mcard-icon">{icon}</div>
                <p class="mcard-lbl">{lbl}</p>
                <p class="mcard-val">{val}</p>
                <p class="mcard-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    DARK_BG = T["chbg"]; CARD_BG = T["chcard"]
    GREEN = "#22c55e"; GREEN2 = "#4ade80"
    GRAY = T["chgray"]; TEXT = T["chtext"]; TEXT2 = T["chtext2"]

    plt.rcParams.update({
        'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
        'axes.edgecolor': GRAY, 'axes.labelcolor': TEXT,
        'xtick.color': TEXT2, 'ytick.color': TEXT2,
        'text.color': TEXT, 'grid.color': GRAY, 'grid.linewidth': 0.5,
    })

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown('<p class="slabel" style="margin-bottom:10px;">Package Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        n, bins, patches = ax.hist(df['Package'], bins=28, edgecolor='none', alpha=0.85)
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.Greens(0.4 + 0.55 * i / len(patches)))
        ax.axvline(df['Package'].mean(), color=GREEN2, linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f"Mean: {df['Package'].mean():.1f} LPA")
        ax.set_xlabel("Package (LPA)", fontsize=10, color=TEXT)
        ax.set_ylabel("Frequency", fontsize=10, color=TEXT)
        ax.set_title("Package Distribution", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    with ch2:
        st.markdown('<p class="slabel" style="margin-bottom:10px;">CGPA vs Package</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        ax.scatter(df['CGPA'], df['Package'], c=df['Package'], cmap='Greens', alpha=0.55, s=22, edgecolors='none')
        z = np.polyfit(df['CGPA'], df['Package'], 1); p = np.poly1d(z)
        x_line = np.linspace(df['CGPA'].min(), df['CGPA'].max(), 100)
        ax.plot(x_line, p(x_line), color=GREEN2, linewidth=2, alpha=0.9, label="Trend Line")
        ax.set_xlabel("CGPA", fontsize=10, color=TEXT); ax.set_ylabel("Package (LPA)", fontsize=10, color=TEXT)
        ax.set_title("CGPA vs Package (Linear Trend)", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT)
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.25)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    ch3, ch4 = st.columns(2)
    with ch3:
        st.markdown('<p class="slabel" style="margin-bottom:10px;">DSA Problems vs Package</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        ax.scatter(df['DSA_Solved'], df['Package'], c=df['CGPA'], cmap='Greens', alpha=0.5, s=20, edgecolors='none')
        z = np.polyfit(df['DSA_Solved'], df['Package'], 1); p = np.poly1d(z)
        x_line = np.linspace(df['DSA_Solved'].min(), df['DSA_Solved'].max(), 100)
        ax.plot(x_line, p(x_line), color=GREEN2, linewidth=2, alpha=0.9, label="Trend")
        ax.set_xlabel("DSA Problems Solved", fontsize=10, color=TEXT)
        ax.set_ylabel("Package (LPA)", fontsize=10, color=TEXT)
        ax.set_title("DSA Solved vs Package", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT)
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.25)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    with ch4:
        st.markdown('<p class="slabel" style="margin-bottom:10px;">Feature Correlation Matrix</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        corr = df.corr()
        cmap2 = mcolors.LinearSegmentedColormap.from_list("gg", ["#0d2818", "#22c55e", "#4ade80"])
        im = ax.imshow(corr, cmap=cmap2, aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        labels2 = ['CGPA','Projects','Internships','DSA','Certs','Comm','Package']
        ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(labels2, rotation=35, ha='right', fontsize=8, color=TEXT2)
        ax.set_yticklabels(labels2, fontsize=8, color=TEXT2)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7,
                        color='white' if abs(corr.iloc[i,j]) > 0.4 else TEXT2)
        ax.set_title("Correlation Matrix", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    ch5, ch6 = st.columns(2)
    with ch5:
        st.markdown('<p class="slabel" style="margin-bottom:10px;">Avg Package by Internships</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        grp = df.groupby('Internships')['Package'].mean()
        colors_bar = [plt.cm.Greens(0.3 + 0.65 * i / max(len(grp)-1, 1)) for i in range(len(grp))]
        bars2 = ax.bar(grp.index.astype(str), grp.values, color=colors_bar, edgecolor='none', width=0.6)
        for bar, val in zip(bars2, grp.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=9, color=GREEN2)
        ax.set_xlabel("Number of Internships", fontsize=10, color=TEXT)
        ax.set_ylabel("Average Package (LPA)", fontsize=10, color=TEXT)
        ax.set_title("Avg Package by Internship Count", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.25)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    with ch6:
        st.markdown('<p class="slabel" style="margin-bottom:10px;">Communication Score vs Package</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        sc_scores = sorted(df['Communication_Score'].unique())
        data_boxes = [df[df['Communication_Score'] == s]['Package'].values for s in sc_scores]
        bp = ax.boxplot(data_boxes, patch_artist=True,
                        medianprops=dict(color=GREEN2, linewidth=2),
                        whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY),
                        flierprops=dict(marker='o', color=GRAY, markersize=3, alpha=0.4))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(plt.cm.Greens(0.3 + 0.55 * i / max(len(sc_scores)-1,1)))
            patch.set_alpha(0.8)
        ax.set_xticklabels([str(s) for s in sc_scores], color=TEXT2, fontsize=8)
        ax.set_xlabel("Communication Score", fontsize=10, color=TEXT)
        ax.set_ylabel("Package (LPA)", fontsize=10, color=TEXT)
        ax.set_title("Package by Communication Score", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.25)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="slabel" style="margin-bottom:4px;">Raw Dataset Preview</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="ssub">First 20 rows of the training data</p>', unsafe_allow_html=True)
    st.dataframe(
        df.head(20).style
        .background_gradient(subset=['Package'], cmap='Greens')
        .format({'CGPA': '{:.2f}', 'Package': '{:.1f}'}),
        use_container_width=True, height=340
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 – MODEL                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_model:
    st.markdown("""
    <div class="anim">
        <p class="slabel">Model Performance</p>
        <p class="stitle">Linear Regression – Evaluation Report</p>
        <p class="ssub">Full analysis of the trained model: metrics, coefficients, and prediction accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

    m = metrics

    mc1, mc2, mc3, mc4 = st.columns(4)
    cards = [
        ("📈", "R² Score",           f"{m['r2']:.4f}",   "Coefficient of determination"),
        ("📉", "Mean Abs Error",     f"{m['mae']:.3f} L", "Average prediction error (LPA)"),
        ("📊", "Mean Squared Error", f"{m['mse']:.3f}",   "Squared prediction error"),
        ("📐", "Root MSE",           f"{m['rmse']:.3f} L","Root of mean squared error"),
    ]
    for col, (icon, lbl, val, sub) in zip([mc1, mc2, mc3, mc4], cards):
        with col:
            st.markdown(f"""
            <div class="mcard anim">
                <div class="mcard-icon">{icon}</div>
                <p class="mcard-lbl">{lbl}</p>
                <p class="mcard-val">{val}</p>
                <p class="mcard-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    lc, rc = st.columns([1, 1.3], gap="large")

    with lc:
        st.markdown('<p class="slabel" style="margin-bottom:14px;">Feature Coefficients</p>', unsafe_allow_html=True)
        coefs = m['coefs']
        max_coef = max(abs(v) for v in coefs.values()) or 1
        friendly = {
            'CGPA': 'CGPA', 'Projects': 'Projects',
            'Internships': 'Internships', 'DSA_Solved': 'DSA Solved',
            'Certifications': 'Certifications', 'Communication_Score': 'Comm. Score'
        }
        rows_html = ""
        for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            pct = abs(coef) / max_coef * 100
            sign = "+" if coef >= 0 else "−"
            rows_html += f"""
            <div class="coef-row">
                <span class="coef-name">{friendly.get(feat, feat)}</span>
                <div class="coef-bar-wrap"><div class="coef-bar-inner" style="width:{pct:.1f}%;"></div></div>
                <span class="coef-val">{sign}{abs(coef):.3f}</span>
            </div>"""
        st.markdown(f'<div class="card anim">{rows_html}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card anim d2" style="margin-top:16px;">
            <p class="slabel" style="margin-bottom:10px;">Model Summary</p>
            <div class="coef-row"><span class="coef-name">Intercept (β₀)</span><span class="coef-val">{m['intercept']:.3f}</span></div>
            <div class="coef-row"><span class="coef-name">Training Samples</span><span class="coef-val" style="color:{T['t2']};">{m['train']}</span></div>
            <div class="coef-row"><span class="coef-name">Test Samples</span><span class="coef-val" style="color:{T['t2']};">{m['test']}</span></div>
            <div class="coef-row"><span class="coef-name">Train / Test Split</span><span class="coef-val" style="color:{T['t2']};">80/20</span></div>
            <div class="coef-row"><span class="coef-name">Algorithm</span><span class="coef-val" style="color:{T['t2']};font-size:11px;">Lin. Reg.</span></div>
        </div>
        """, unsafe_allow_html=True)

    with rc:
        DARK_BG = T["chbg"]; CARD_BG = T["chcard"]
        GREEN = "#22c55e"; GREEN2 = "#4ade80"
        GRAY = T["chgray"]; TEXT = T["chtext"]; TEXT2 = T["chtext2"]

        plt.rcParams.update({
            'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
            'axes.edgecolor': GRAY, 'axes.labelcolor': TEXT,
            'xtick.color': TEXT2, 'ytick.color': TEXT2,
            'text.color': TEXT, 'grid.color': GRAY, 'grid.linewidth': 0.5,
        })

        st.markdown('<p class="slabel" style="margin-bottom:10px;">Actual vs Predicted Packages</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        ax.scatter(m['y_test'], m['y_pred'], c=m['y_pred'],
                   cmap='Greens', alpha=0.65, s=28, edgecolors='none', label='Predictions')
        lims = [min(m['y_test'].min(), m['y_pred'].min()) - 1,
                max(m['y_test'].max(), m['y_pred'].max()) + 1]
        ax.plot(lims, lims, '--', color=GREEN2, linewidth=1.5, alpha=0.7, label='Perfect Prediction')
        ax.set_xlabel("Actual Package (LPA)", fontsize=10, color=TEXT)
        ax.set_ylabel("Predicted Package (LPA)", fontsize=10, color=TEXT)
        ax.set_title(f"Actual vs Predicted  |  R² = {m['r2']:.4f}", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT)
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.25)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown('<p class="slabel" style="margin-bottom:10px;">Residuals (Error) Distribution</p>', unsafe_allow_html=True)
        residuals = m['y_test'] - m['y_pred']
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        n2, bins2, patches2 = ax.hist(residuals, bins=28, edgecolor='none', alpha=0.85)
        for i, patch in enumerate(patches2):
            patch.set_facecolor(plt.cm.Greens(0.3 + 0.65 * i / len(patches2)))
        ax.axvline(0, color=GREEN2, linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Error')
        ax.set_xlabel("Residual (Actual − Predicted)", fontsize=10, color=TEXT)
        ax.set_ylabel("Frequency", fontsize=10, color=TEXT)
        ax.set_title("Residuals Distribution", fontsize=12, fontweight='bold', color=TEXT, pad=12)
        ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.25)
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    r2_pct   = int(m['r2'] * 100)
    r2_color = T["g4"] if m['r2'] > 0.85 else ("#fbbf24" if m['r2'] > 0.70 else "#f87171")
    r2_label = "Excellent" if m['r2'] > 0.85 else ("Good" if m['r2'] > 0.70 else "Needs Improvement")

    st.markdown(f"""
    <div class="card anim" style="text-align:center;padding:36px 28px;">
        <p class="slabel" style="margin-bottom:8px;">Overall Model Quality</p>
        <p style="font-family:'Space Grotesk',sans-serif;font-size:64px;font-weight:800;
                  color:{r2_color};line-height:1;margin:0;letter-spacing:-2px;">{r2_pct}%</p>
        <p style="font-size:16px;color:{T['t2']};margin:10px 0 0 0;">
            R² Score → <strong style="color:{r2_color};">{r2_label}</strong> &nbsp;|&nbsp;
            The model explains <strong style="color:{r2_color};">{r2_pct}%</strong> of package variance.
        </p>
        <div class="ptrack" style="max-width:480px;margin:18px auto 0;height:8px;">
            <div class="pbar" style="width:{r2_pct}%;background:linear-gradient(90deg,{T['g6']},{r2_color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)