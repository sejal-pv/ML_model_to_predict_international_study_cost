import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load trained model
model = joblib.load("model.pkl")

# App Title and Description
st.markdown("<h1 style='color:#4B8BBE'>🎓 Study Abroad Cost Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>Fill in your education preferences to estimate the total cost and visualize your input compared to other students.</p>", unsafe_allow_html=True)

st.markdown("### 📝 User Input")
duration = st.number_input("⏳ Duration (Years)", min_value=0.6, max_value=5.0)
tuition = st.number_input("💵 Tuition (USD)", min_value=0.0, max_value=62000.0)
rent = st.number_input("🛏️ Rent (USD)", min_value=150.0, max_value=2500.0)
visa_fee = st.number_input("🛂 Visa Fee (USD)", min_value=40.0, max_value=490.0)
insurance = st.number_input("🩺 Insurance (USD)", min_value=200.0, max_value=1500.0)
exchange_rate = st.number_input("💱 Exchange Rate", min_value=0.0, max_value=42150.0)
living_cost_index = st.number_input("🏠 Living Cost Index", min_value=27.0, max_value=122.0)
submitted = st.button("Predict")

# If form submitted, process input and predict
if 'submitted' in locals() and submitted:
    input_df = pd.DataFrame({
        'Duration_Years': [duration],
        'Tuition_USD': [tuition],
        'Living_Cost_Index': [living_cost_index],
        'Rent_USD': [rent],
        'Visa_Fee_USD': [visa_fee],
        'Insurance_USD': [insurance],
        'Exchange_Rate': [exchange_rate]
    })

    try:
        prediction = model.predict(input_df)[0]
      
        st.markdown("### 🎯 Prediction Result")
        st.success(f"Estimated Total Annual Cost: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # --- 1. Duration_Years - Bar Chart ---
    if 'Duration_Years' in input_df.columns:
        st.markdown("### ⏳ Duration (Years) - Bar Chart")
        fig1, ax1 = plt.subplots()
        duration_counts = input_df['Duration_Years'].value_counts().sort_index()
        sns.barplot(x=duration_counts.index, y=duration_counts.values, ax=ax1, color='skyblue')
        ax1.axvline(input_df['Duration_Years'][0], color='red', linestyle='--', label='User Input')
        ax1.set_ylabel("Frequency")
        ax1.set_xlabel("Years")
        ax1.set_title("Program Duration Distribution")
        ax1.legend()
        st.pyplot(fig1)

    # --- 2. Tuition_USD - Pie Chart ---
    if 'Tuition_USD' in input_df.columns:
        st.markdown("### 💵 Tuition Fee - Pie Chart")
        tuition_value = input_df['Tuition_USD'][0]
        remaining = max(1, input_df['Tuition_USD'].mean() * 1.5 - tuition_value)  # Simulated comparison
        tuition_parts = pd.Series({
            'User Tuition': tuition_value,
            'Remaining Budget': remaining
        })
        fig2, ax2 = plt.subplots()
        ax2.pie(tuition_parts, labels=tuition_parts.index, autopct='%1.1f%%', startangle=140)
        ax2.set_title("Tuition Cost Breakdown")
        ax2.axis('equal')
        st.pyplot(fig2)

    # --- 3. Rent, Visa, Exchange Rate - Line Chart ---
    line_features = ['Rent_USD', 'Visa_Fee_USD', 'Exchange_Rate']
    existing_line_features = [col for col in line_features if col in input_df.columns]

    if existing_line_features:
        st.markdown("### 📈 Rent, Visa & Exchange Rate - Line Chart")
        fig3, ax3 = plt.subplots()
        for col in existing_line_features:
            data = input_df[col].sort_values().reset_index(drop=True)
            ax3.plot(data, label=col)
            user_val = input_df[col][0]
            ax3.axhline(user_val, linestyle='--', linewidth=1.5, label=f'{col} (User)', alpha=0.6)
        ax3.set_title("Line Graph of Rent, Visa Fee, Exchange Rate")
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("Value")
        ax3.legend()
        st.pyplot(fig3)
