import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("models/lung_cancer_model.pkl")

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

st.title("🫁 Lung Cancer Detection System")
st.markdown("Enter patient details to predict lung cancer risk.")

# import pandas as pd

# Get column names from model (safer method)
feature_names = model.estimators_[0].feature_names_in_

input_dict = {}

st.write("Enter patient details:")

for feature in feature_names:
    if feature == "AGE":
        input_dict[feature] = st.number_input(feature, min_value=1, max_value=120, value=30)
    else:
        input_dict[feature] = st.selectbox(f"{feature} (0 = No, 1 = Yes)", [0, 1])

if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Lung Cancer\nProbability: {probability:.2f}")
    else:
       st.success(f"✅ Low Risk of Lung Cancer\nProbability: {probability*100:.2f}%")