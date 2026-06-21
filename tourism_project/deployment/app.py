import os
import pandas as pd
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🌍")
st.title("Visit with Us - Wellness Package Purchase Prediction")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "jaiwala/tourism-package-model")
HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model():
    if HF_MODEL_REPO:
        model_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="best_model.joblib",
            repo_type="model",
            token=HF_TOKEN,
        )
    else:
        model_file = "tourism_project/model_building/best_model.joblib"
    return joblib.load(model_file)

st.subheader("Enter customer details")

input_data = {
    "Age": st.number_input("Age", min_value=18, max_value=90, value=35),
    "TypeofContact": st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"]),
    "CityTier": st.selectbox("City Tier", [1, 2, 3]),
    "Occupation": st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "NumberOfPersonVisiting": st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2),
    "PreferredPropertyStar": st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5]),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "NumberOfTrips": st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=3),
    "Passport": st.selectbox("Passport", [0, 1]),
    "OwnCar": st.selectbox("Own Car", [0, 1]),
    "NumberOfChildrenVisiting": st.number_input("Children Visiting (<5 years)", min_value=0, max_value=6, value=0),
    "Designation": st.selectbox("Designation", ["Manager", "Senior Manager", "AVP", "VP", "Executive"]),
    "MonthlyIncome": st.number_input("Monthly Income", min_value=1000, max_value=500000, value=50000),
    "PitchSatisfactionScore": st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3),
    "ProductPitched": st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]),
    "NumberOfFollowups": st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2),
    "DurationOfPitch": st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=200, value=30),
}

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])  # required rubric step: save inputs into a dataframe
    st.write("Input DataFrame", input_df)

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.info("Check HF_MODEL_REPO / HF_TOKEN in Space settings and confirm best_model.joblib exists.")
        st.stop()

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction (ProdTaken): {int(pred)}")
    st.info(f"Probability of purchase: {prob:.2%}")
