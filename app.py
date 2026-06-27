import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Titanic Survivor Prediction",
    page_icon="🚢",
    layout="centered",
)


# Custom CSS 

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Deep ocean gradient backdrop with soft light-ray accents */
        .stApp {
            background: radial-gradient(circle at 20% 0%, rgba(45, 130, 190, 0.25), transparent 45%),
                        radial-gradient(circle at 85% 15%, rgba(80, 200, 220, 0.15), transparent 40%),
                        linear-gradient(160deg, #061626 0%, #0b2540 45%, #0f3a5f 100%);
            background-attachment: fixed;
        }

        /* Hide default Streamlit chrome for a cleaner look */
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        /* Hero title block */
        .hero-wrap { text-align: center; margin-top: 0.5rem; margin-bottom: 1.8rem; }
        .hero-emoji {
            font-size: 2.6rem;
            display: inline-block;
            animation: bob 3.2s ease-in-out infinite;
        }
        @keyframes bob {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }
        .hero-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 2.1rem;
            color: #eaf4fb;
            margin: 0.3rem 0 0.2rem 0;
            letter-spacing: 0.3px;
        }
        .hero-sub {
            font-family: 'Inter', sans-serif;
            color: #9fc3dd;
            font-size: 0.98rem;
            max-width: 480px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* Frosted glass card */
        div.st-key-glass_card {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 22px;
            padding: 1.8rem 1.6rem 1.2rem 1.6rem;
            backdrop-filter: blur(22px) saturate(160%);
            -webkit-backdrop-filter: blur(22px) saturate(160%);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.2rem;
        }

        div.st-key-result_card_survive {
            background: rgba(46, 196, 142, 0.14);
            border: 1px solid rgba(46, 196, 142, 0.4);
            border-radius: 22px;
            padding: 1.4rem 1.6rem;
            backdrop-filter: blur(22px) saturate(160%);
            -webkit-backdrop-filter: blur(22px) saturate(160%);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        }

        div.st-key-result_card_perish {
            background: rgba(220, 90, 90, 0.14);
            border: 1px solid rgba(220, 90, 90, 0.4);
            border-radius: 22px;
            padding: 1.4rem 1.6rem;
            backdrop-filter: blur(22px) saturate(160%);
            -webkit-backdrop-filter: blur(22px) saturate(160%);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        }

        /* Section labels inside the glass card */
        .section-label {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #cfe8f7;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
            letter-spacing: 0.2px;
        }

        /* Inputs — soften the default Streamlit widget chrome */
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            border-radius: 12px !important;
        }
        div[data-baseweb="select"] * {
            color: #eaf4fb !important;
            -webkit-text-fill-color: #eaf4fb !important;
        }
        div[data-baseweb="select"] svg {
            fill: #eaf4fb !important;
        }
        label, .stMarkdown p { color: #d8ebf6 !important; }

        /* Number input — reset every nested element's background first (BaseWeb
           nests several divs and the grey one isn't always the outer wrapper),
           then repaint only the parts we want styled. */
        div[data-testid="stNumberInput"] * {
            background-color: transparent !important;
        }
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: rgba(255, 255, 255, 0.07) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }
        div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
            border-color: rgba(37, 198, 160, 0.7) !important;
            box-shadow: 0 0 0 1px rgba(37, 198, 160, 0.35) !important;
        }
        div[data-testid="stNumberInput"] input {
            color: #eaf4fb !important;
            -webkit-text-fill-color: #eaf4fb !important;
            caret-color: #eaf4fb !important;
        }
        div[data-testid="stNumberInput"] button {
            background: rgba(255, 255, 255, 0.1) !important;
            border: none !important;
        }
        div[data-testid="stNumberInput"] button svg {
            fill: #eaf4fb !important;
        }
        div[data-testid="stNumberInput"] button:hover {
            background: rgba(255, 255, 255, 0.22) !important;
        }

        /* Override Streamlit's theme accent variable directly — the slider's
           track fill and value label pull from this, not just the thumb,
           so a variable override is more reliable than chasing nested divs */
        :root, .stApp {
            --primary-color: #25c6a0 !important;
        }

        /* Slider — Streamlit colors the track fill and thumb via inline
           style attributes. Target only elements that actually carry an
           inline style, rather than every div, so structural/shape divs
           (like the round thumb) aren't disturbed. */
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style] {
            background: #25c6a0 !important;
            border-color: #25c6a0 !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #25c6a0 !important;
            border-color: #25c6a0 !important;
        }
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
            color: #9fc3dd !important;
        }
        div[data-testid="stThumbValue"],
        div[data-testid="stSliderThumbValue"] {
            color: #25c6a0 !important;
        }

        /* Predict button — gradient accent, the one bold signature element.
           Target the testid wrapper so width:100% actually reaches the <button>. */
        div[data-testid="stButton"] {
            width: 100%;
        }
        div[data-testid="stButton"] button {
            width: 100%;
            border-radius: 14px;
            border: none;
            background: linear-gradient(135deg, #1f8ecf 0%, #25c6a0 100%);
            color: white !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.02rem;
            padding: 0.7rem 1rem;
            white-space: nowrap;
            box-shadow: 0 6px 18px rgba(31, 142, 207, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stButton"] button p {
            color: white !important;
            font-weight: 600;
        }
        div[data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(31, 142, 207, 0.45);
        }

        .footer-note {
            text-align: center;
            color: #7ea7c4;
            font-size: 0.82rem;
            margin-top: 1.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load saved model artifacts

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    return model, scaler, feature_columns, encoders


model, scaler, feature_columns, encoders = load_artifacts()

# Hero

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-emoji">🚢</div>
        <div class="hero-title">Titanic Survivor Prediction</div>
        <div class="hero-sub">
            Enter a passenger's details and a tuned Random Forest model
            (82% test accuracy) will estimate their odds of survival.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input form, inside a frosted glass card

with st.container(key="glass_card"):
    st.markdown('<div class="section-label">Passenger Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)
        title = st.selectbox(
            "Title", ["Mr", "Mrs", "Ms", "Dr", "Rev", "Military", "Nobility"]
        )
        embarked = st.selectbox(
            "Port of Embarkation",
            ["S - Southampton", "C - Cherbourg", "Q - Queenstown"],
        )

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Total Fare Paid (for the ticket)", min_value=0.0, max_value=600.0, value=32.0)
        tkt_count = st.number_input("People sharing this ticket (incl. self)", min_value=1, max_value=10, value=1)
        cabin_deck = st.selectbox("Cabin Deck (M = unknown)", ["M", "A", "B", "C", "D", "E", "F", "G"])

    predict_clicked = st.button("Predict Survival")

# Prediction logic — unchanged from the original pipeline

if predict_clicked:
    embarked_code = embarked.split(" - ")[0]
    fare_per_ticket = fare / tkt_count
    num_family = sibsp + parch + 1

    age_bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    fare_bins = [0, 20, 40, 60, 80, 150]

    age_bin = pd.cut([age], bins=age_bins)[0]
    fare_for_bin = min(fare_per_ticket, 149.99)  # clip to stay within trained bin range
    fare_bin = pd.cut([fare_for_bin], bins=fare_bins)[0]

    row = {
        "Pclass": pclass,
        "Sex": sex,
        "SibSp": sibsp,
        "Parch": parch,
        "Cabin": cabin_deck,
        "Embarked": embarked_code,
        "Titles": title,
        "Age_Bin": str(age_bin),
        "Fare_Bin": str(fare_bin),
        "Num_Family": num_family,
    }

    input_df = pd.DataFrame([row]).astype(str)

    # Apply the same fitted LabelEncoders used during training
    for col, le in encoders.items():
        if col in input_df.columns:
            val = input_df.loc[0, col]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0  # fallback for an unseen category

    # One-hot encode the same columns as training (now on their label-encoded values)
    one_hot_input = pd.get_dummies(input_df, columns=["Sex", "Embarked", "Titles"])
    one_hot_input = one_hot_input.astype(int)

    # Align to the exact training column order, filling any missing dummy columns with 0
    one_hot_input = one_hot_input.reindex(columns=feature_columns, fill_value=0)

    # Scale and predict
    scaled_input = scaler.transform(one_hot_input.values)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        with st.container(key="result_card_survive"):
            st.markdown(
                f"""
                <div class="section-label" style="color:#9be8cf;">Prediction</div>
                <div style="font-size:1.4rem; font-weight:700; color:#eaf4fb; font-family:'Outfit',sans-serif;">
                    🎉 Likely to survive
                </div>
                <div style="color:#cfe8f7; margin-top:0.3rem;">
                    Estimated survival probability: <b>{probability:.1%}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        with st.container(key="result_card_perish"):
            st.markdown(
                f"""
                <div class="section-label" style="color:#f3b0b0;">Prediction</div>
                <div style="font-size:1.4rem; font-weight:700; color:#eaf4fb; font-family:'Outfit',sans-serif;">
                    💀 Unlikely to survive
                </div>
                <div style="color:#cfe8f7; margin-top:0.3rem;">
                    Estimated survival probability: <b>{probability:.1%}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footer-note">Made By Rehan Shaikh</div>',
    unsafe_allow_html=True,
)