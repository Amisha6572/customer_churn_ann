import streamlit as st
import numpy as np
import pandas as pd
import pickle
import keras

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- base ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b27 60%, #1a1f35 100%);
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
* { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ---------- typography ---------- */
h1 { font-size: 2rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg, #58a6ff, #a371f7);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { color: #e6edf3 !important; }
p, li, span, label { color: #c9d1d9 !important; }

/* ---------- section card ---------- */
.card {
    background: #161b27;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #58a6ff !important;
    margin-bottom: 16px;
}

/* ---------- inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] input { color: #e6edf3 !important; }
div[role="listbox"] { background: #161b27 !important; border: 1px solid #30363d !important; }
div[role="option"] { color: #c9d1d9 !important; }
div[role="option"]:hover { background: #21262d !important; }

/* ---------- slider ---------- */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #58a6ff, #a371f7) !important;
}

/* ---------- radio ---------- */
[data-testid="stRadio"] label { color: #c9d1d9 !important; }

/* ---------- primary button ---------- */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #238636, #2ea043) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    letter-spacing: 0.03em;
    transition: filter 0.2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    filter: brightness(1.15);
    box-shadow: 0 0 18px rgba(46,160,67,0.45) !important;
}

/* ---------- secondary button ---------- */
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
    background: #161b27;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] p { color: #8b949e !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 2rem !important; font-weight: 800 !important; }

/* ---------- alerts ---------- */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ---------- expander ---------- */
[data-testid="stExpander"] { background: #161b27 !important; border: 1px solid #21262d !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #c9d1d9 !important; }

/* ---------- dataframe ---------- */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---------- divider ---------- */
hr { border-color: #21262d !important; }

/* ---------- sidebar labels ---------- */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #8b949e !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] h3 { color: #e6edf3 !important; font-size: 0.95rem !important; }

/* ---------- progress bar ---------- */
.risk-bar-wrap {
    background: #21262d;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin: 8px 0 4px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# ASSET LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    try:
        # Load from TF SavedModel directory — avoids Keras 3 config
        # deserialization bugs (InputLayer batch_shape / Dense config issues)
        import tensorflow as tf
        model = tf.saved_model.load("churn_model_saved")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_err = load_assets()


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ ChurnGuard AI")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "A neural network trained on 10,000 bank customers to predict "
        "churn probability. Accuracy: **85.75%**."
    )
    st.markdown("---")
    st.markdown("**Model Architecture**")
    st.markdown(
        "- Input: 11 features  \n"
        "- Dense 64 → Dropout  \n"
        "- Dense 32 → Dropout  \n"
        "- Dense 16  \n"
        "- Output: Sigmoid"
    )
    st.markdown("---")
    st.markdown("**Threshold**")
    threshold = st.slider("Churn threshold", 0.30, 0.80, 0.50, 0.01,
                          help="Probability above this value is flagged as churn risk.")
    st.markdown("---")
    st.caption("Built with Streamlit · Keras 3 · TF 2.21")


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("<h1>🛡️ ChurnGuard AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8b949e;font-size:1rem;margin-top:-8px;'>"
    "Neural Network · Customer Churn Prediction · Bank Retention Analytics"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if load_err:
    st.error(f"**Model failed to load:** {load_err}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# INPUT FORM  (3 columns)
# ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="card-title">👤 Demographics</div>', unsafe_allow_html=True)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender    = st.selectbox("Gender", ["Male", "Female"])
    age       = st.slider("Age", 18, 92, 38)

with col2:
    st.markdown('<div class="card-title">💳 Financial Profile</div>', unsafe_allow_html=True)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850,
                                   value=650, step=1)
    balance      = st.number_input("Account Balance ($)", min_value=0.0,
                                   value=75_000.0, step=500.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0,
                                       value=60_000.0, step=500.0, format="%.2f")

with col3:
    st.markdown('<div class="card-title">📈 Engagement</div>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (years)", 0, 10, 4)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card     = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    is_active       = st.radio("Active Member?",   ["Yes", "No"], horizontal=True)

st.markdown("")  # spacer

predict_btn = st.button("⚡  Analyze Churn Risk", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PREPROCESSING HELPER
# ──────────────────────────────────────────────────────────────
def build_input():
    gender_enc  = 1 if gender == "Male" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain   = 1 if geography == "Spain" else 0
    cr_card_val = 1 if has_cr_card == "Yes" else 0
    active_val  = 1 if is_active == "Yes" else 0
    return np.array([[
        credit_score, gender_enc, age, tenure, balance,
        num_of_products, cr_card_val, active_val,
        estimated_salary, geo_germany, geo_spain
    ]])


# ──────────────────────────────────────────────────────────────
# PREDICTION & RESULTS
# ──────────────────────────────────────────────────────────────
if predict_btn:
    try:
        import tensorflow as tf
        raw    = build_input()
        scaled = scaler.transform(raw).astype("float32")
        tensor = tf.constant(scaled)
        prob   = float(model.serve(tensor)[0][0])
        is_churn = prob >= threshold

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        # ── KPI row ──────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Churn Probability",  f"{prob:.1%}")
        k2.metric("Retention Probability", f"{1 - prob:.1%}")
        k3.metric("Risk Threshold",     f"{threshold:.0%}")
        k4.metric("Verdict", "⚠️ High Risk" if is_churn else "✅ Stable")

        # ── Visual risk bar ───────────────────────────────────
        bar_color = (
            "#da3633" if prob >= 0.70 else
            "#e3b341" if prob >= threshold else
            "#2ea043"
        )
        st.markdown(
            f"""
            <div style="margin:16px 0 4px;">
                <span style="color:#8b949e;font-size:0.8rem;font-weight:600;
                             text-transform:uppercase;letter-spacing:0.06em;">
                    Risk Level
                </span>
            </div>
            <div class="risk-bar-wrap">
                <div class="risk-bar-fill"
                     style="width:{prob*100:.1f}%;background:{bar_color};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        color:#8b949e;font-size:0.75rem;margin-top:2px;">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Verdict banner ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if is_churn:
            st.error(
                f"**High Churn Risk — {prob:.1%} probability**  \n"
                "This customer shows strong indicators of leaving. "
                "Consider proactive retention: personalised offers, "
                "loyalty rewards, or a dedicated account manager outreach."
            )
        else:
            st.success(
                f"**Low Churn Risk — {prob:.1%} probability**  \n"
                "This customer appears stable and engaged. "
                "Standard retention practices should be sufficient."
            )

        # ── Feature summary ───────────────────────────────────
        with st.expander("🔍 Input Summary"):
            summary = pd.DataFrame({
                "Feature": [
                    "Geography", "Gender", "Age", "Tenure",
                    "Credit Score", "Balance", "Estimated Salary",
                    "Products", "Credit Card", "Active Member",
                ],
                "Value": [
                    geography, gender, age, tenure,
                    credit_score, f"${balance:,.0f}", f"${estimated_salary:,.0f}",
                    num_of_products, has_cr_card, is_active,
                ],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────────
# DATASET PREVIEW
# ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📂 Training Dataset Preview"):
    try:
        df = pd.read_csv("Churn Modeling.csv")
        total    = len(df)
        churned  = int(df["Exited"].sum())
        retained = total - churned

        d1, d2, d3 = st.columns(3)
        d1.metric("Total Records",    f"{total:,}")
        d2.metric("Churned",          f"{churned:,}",  f"{churned/total:.1%}")
        d3.metric("Retained",         f"{retained:,}", f"{retained/total:.1%}")

        st.dataframe(df.head(10), use_container_width=True)
    except FileNotFoundError:
        st.info("Place `Churn Modeling.csv` in the project folder to preview the dataset.")
