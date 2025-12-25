import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="AnkushWaghmare/Tourism-Project-model",
    filename="best_Tourism-Project_model_v1.joblib"
)

# Load the trained model
model = joblib.load(model_path)

# Streamlit UI
st.title("MLOPS – Customer Package Purchase Prediction App")
st.write(
    "This internal application predicts whether a customer is likely to "
    "purchase a travel package based on demographic and interaction details."
)
st.write("Please enter the customer details below.")

# -----------------------------
# Customer Details
# -----------------------------
Age = st.number_input("Age", min_value=18, max_value=100, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox("City Tier", [1, 2, 3])

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Freelancer", "Small Business", "Large Business"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1, max_value=10, value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips (per year)",
    min_value=0, max_value=50, value=2
)

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0, max_value=5, value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=5000, max_value=500000, value=50000
)

# -----------------------------
# Interaction Details
# -----------------------------
PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1, max_value=5, value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0, max_value=20, value=2
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=1, max_value=120, value=15
)

# -----------------------------
# Prepare input data
# -----------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# Classification threshold
classification_threshold = 0.5

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    if prediction == 1:
        st.success("✅ The customer is likely to purchase the package.")
    else:
        st.error("❌ The customer is unlikely to purchase the package.")
