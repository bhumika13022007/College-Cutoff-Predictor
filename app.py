
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="College Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

# ==========================================
# 2. SIDEBAR + THEME
# ==========================================
with st.sidebar:
    st.markdown("## 🎓 Cutoff Predictor")
    st.caption("SMART ADMISSION ANALYTICS")
    st.divider()

    dark_mode = st.checkbox(
        "🌙 Dark Mode",
        value=True,
        key="dark_mode"
    )

    st.divider()
    st.markdown("### About")
    st.write(
        "Estimate college admission cutoffs using "
        "your admission preferences and our trained ML model."
    )

    st.info("🤖 Model: Random Forest Regressor")
    st.caption("Powered by Python & Streamlit")

# ==========================================
# 3. PROFESSIONAL UI THEME
# ==========================================
if dark_mode:
    bg = "#0B1020"
    panel = "#141C30"
    input_bg = "#1B2640"
    text = "#F1F5F9"
    muted = "#A5B4CC"
    border = "#2A3854"
    accent = "#8B9CFF"
    accent2 = "#39D6C0"
    result_bg = "#142B32"
else:
    bg = "#F3F6FC"
    panel = "#FFFFFF"
    input_bg = "#F8FAFF"
    text = "#17233D"
    muted = "#66748D"
    border = "#DFE6F1"
    accent = "#4F46E5"
    accent2 = "#0891B2"
    result_bg = "#E8F8F3"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text};
    }}

    [data-testid="stHeader"] {{
        background: {bg};
    }}

    [data-testid="stSidebar"] {{
        background: {panel};
        border-right: 1px solid {border};
    }}

    [data-testid="stSidebar"] * {{
        color: {text};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }}

    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {{
        color: {text};
    }}

    .hero {{
        background: linear-gradient(
            125deg, {panel}, {input_bg}
        );
        border: 1px solid {border};
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
    }}

    .eyebrow {{
        color: {accent2};
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }}

    .hero-title {{
        color: {text};
        font-size: clamp(27px, 4vw, 40px);
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 12px;
    }}

    .hero-description {{
        color: {muted};
        font-size: 15px;
        line-height: 1.7;
    }}

    .section-card {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
    }}

    [data-testid="stForm"] {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 24px;
    }}

    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label {{
        color: {text} !important;
        font-weight: 600;
    }}

    [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {border} !important;
        border-radius: 10px !important;
        min-height: 44px;
    }}

    [data-baseweb="select"] * {{
        color: {text} !important;
    }}

    [data-baseweb="popover"],
    [data-baseweb="menu"] {{
        background-color: {panel} !important;
    }}

    [role="option"] {{
        background-color: {panel} !important;
        color: {text} !important;
    }}

    [role="option"]:hover {{
        background-color: {input_bg} !important;
    }}

    [data-testid="stNumberInput"] input {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-radius: 10px !important;
    }}

    div.stButton > button,
    div.stFormSubmitButton > button {{
        border-radius: 11px;
        min-height: 48px;
        font-weight: 700;
        transition: all 0.2s ease;
    }}

    div.stFormSubmitButton > button {{
        background: linear-gradient(
            100deg, {accent}, {accent2}
        );
        color: #FFFFFF;
        border: none;
    }}

    div.stFormSubmitButton > button:hover {{
        filter: brightness(1.08);
        border: none;
        transform: translateY(-1px);
    }}

    [data-testid="stMetric"] {{
        background: {result_bg};
        border: 1px solid {border};
        padding: 20px;
        border-radius: 16px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {muted} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {accent2} !important;
        font-weight: 800;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid {border};
        border-radius: 12px;
    }}

    hr {{
        border-color: {border};
    }}

    .footer {{
        text-align: center;
        color: {muted};
        font-size: 13px;
        padding: 18px 0 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi hua. Please check model.pkl.")
    st.code(str(e))
    st.stop()

# ==========================================
# 5. EXTRACT TRAINED INPUT OPTIONS
# ==========================================
try:
    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]

    feature_names = list(model.feature_names_in_)

    expected_features = [
        "university_id",
        "year",
        "category",
        "seat_type",
        "figure_kind"
    ]

    if feature_names != expected_features:
        st.error("Model ke input columns match nahi karte.")
        st.write("Model columns:", feature_names)
        st.stop()

    encoded_features = [
        "university_id",
        "category",
        "seat_type",
        "figure_kind"
    ]

    available_options = {
        feature: list(values)
        for feature, values in zip(
            encoded_features,
            encoder.categories_
        )
    }

    for feature, options in available_options.items():
        if not options:
            st.error(f"{feature} ke liye options available nahi hain.")
            st.stop()

except Exception as e:
    st.error("Model ki input configuration read nahi hui.")
    st.code(str(e))
    st.stop()

# ==========================================
# 6. HERO HEADER
# ==========================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMART ADMISSION ANALYTICS</div>
        <div class="hero-title">
            College Cutoff Predictor 🎓
        </div>
        <div class="hero-description">
            Make informed admission decisions with an ML-based
            estimate of college cutoff rank or score.
            Choose your preferences below to get started.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 7. INPUT FORM
# ==========================================
st.subheader("📋 Admission Details")
st.caption(
    "Select the options available in your trained model."
)

with st.form("cutoff_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        university_id = st.selectbox(
            "🏛️ University ID",
            options=available_options["university_id"],
            format_func=str
        )

        category = st.selectbox(
            "👥 Category",
            options=available_options["category"],
            format_func=str
        )

        year = st.number_input(
            "📅 Admission Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1
        )

    with col2:
        seat_type = st.selectbox(
            "🎓 Seat Type",
            options=available_options["seat_type"],
            format_func=str
        )

        figure_kind = st.selectbox(
            "📊 Prediction Type",
            options=available_options["figure_kind"],
            format_func=str
        )

        st.caption(
            "Choose the available rank or score type."
        )

    st.divider()

    submitted = st.form_submit_button(
        "✨ Predict Cutoff",
        type="primary",
        use_container_width=True
    )

# ==========================================
# 8. PREDICTION
# ==========================================
if submitted:
    input_data = pd.DataFrame(
        [{
            "university_id": university_id,
            "year": year,
            "category": category,
            "seat_type": seat_type,
            "figure_kind": figure_kind
        }],
        columns=feature_names
    )

    try:
        with st.spinner("Analyzing your admission details..."):
            prediction = model.predict(input_data)
            predicted_value = float(prediction[0])

        st.divider()
        st.subheader("📈 Prediction Result")

        result_col, info_col = st.columns([1.2, 1])

        with result_col:
            st.metric(
                label=f"Estimated {str(figure_kind).title()}",
                value=f"{predicted_value:,.2f}"
            )

        with info_col:
            st.success("Prediction generated successfully!")
            st.markdown(
                f"**University ID:** {university_id}"
            )
            st.markdown(
                f"**Admission Year:** {int(year)}"
            )

        with st.expander("🔎 View Complete Input Details"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Note: This is an ML-generated estimate, not a guaranteed "
            "official admission cutoff. Always verify official data."
        )

    except Exception as e:
        st.error("Prediction generate nahi hui.")
        st.code(str(e))

# ==========================================
# 9. FOOTER
# ==========================================
st.divider()

st.markdown(
    """
    <div class="footer">
        College Cutoff Predictor · Python · Scikit-learn ·
        Random Forest · Streamlit
        <br>
        Built to support smarter admission decisions.
    </div>
    """,
    unsafe_allow_html=True
)
