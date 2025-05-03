import streamlit as st
import pickle
import numpy as np
print(np.__version__)

# Load the trained model
with open("Loan_Approval_Prediction.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("Loan Approval Prediction App")
st.subheader("Using Random Forest Model")

# User input
st.write("Please enter the following information:")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
credit_history = st.selectbox("Credit History", ["Has Credit History (1)", "No Credit History (0)"])

# Feature engineering
total_income = applicant_income + coapplicant_income

# Encoding inputs as model expects
gender_male = 1 if gender == "Male" else 0
married_yes = 1 if married == "Yes" else 0
credit_history_val = 1 if credit_history.startswith("Has") else 0

# Arrange features in correct order
input_features = [gender_male, married_yes, total_income, loan_amount, credit_history_val]

# Predict button
if st.button("Check Loan Approval"):
    prediction = model.predict([input_features])[0]
    if prediction == 1:
        st.success("✅ Loan is likely to be Approved.")
    else:
        st.error("❌ Loan is likely to be Denied.")
