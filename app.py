import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("model.pkl")  # Make sure model.pkl exists in the same folder

# Define feature names (ensure they match training features)
feature_names = [
    "Living_Cost_Index",
    "Tuition_USD",
    "Exchange_Rate",
    "Duration_Years",
    "Rent_USD",
    "Visa_Fee_USD",
    "Insurance_USD"
]

# UI Title
st.title("🎓 University Cost Predictor")
st.markdown("This app predicts the **Total Annual Cost (USD)** for studying abroad based on your input features.")

# Collect input for each feature
st.subheader("🔢 Enter the following details:")
user_input = {}
for feature in feature_names:
    label = feature.replace("_", " ")
    user_input[feature] = st.number_input(f"{label}", min_value=0.0, format="%.2f")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict and display results
if st.button("🔍 Predict Total Annual Cost"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Total Annual Cost: **${prediction:,.2f}**")

    # 📊 Plot input values
    st.subheader("📌 Input Feature Values")
    fig1, ax1 = plt.subplots()
    ax1.barh(input_df.columns, input_df.values[0], color="skyblue")
    ax1.set_xlabel("Input Value")
    ax1.set_title("User Input Feature Values")
    st.pyplot(fig1)

    # 📊 Plot feature importance (if available)
    if hasattr(model, "feature_importances_"):
        st.subheader("🌟 Feature Importance (From Model)")
        importance = model.feature_importances_
        fig2, ax2 = plt.subplots()
        ax2.barh(feature_names, importance, color="orange")
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Random Forest Feature Importance")
        st.pyplot(fig2)
    else:
        st.warning("This model does not support feature importance.")
