import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survivor Prediction", page_icon="🚢")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    return model, scaler, feature_columns, encoders


model, scaler, feature_columns, encoders = load_artifacts()

st.title("🚢 Titanic Survivor Prediction")
st.write(
    "Enter passenger details below. The model is a Random Forest Classifier "
    "tuned with RandomizedSearchCV + GridSearchCV (82% test accuracy)."
)

# --- User Inputs ---
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

st.divider()

if st.button("Predict Survival", type="primary"):
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
        st.success(f"🎉 Likely to **survive** — predicted probability: {probability:.1%}")
    else:
        st.error(f"💀 Likely **not** to survive — predicted survival probability: {probability:.1%}")

st.caption("Made By Rehan Shaikh.")