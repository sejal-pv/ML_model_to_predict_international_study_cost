import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 


st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #e0f7fa !important; /* Light blue */
    }
    [data-testid="stSidebar"] * {
        color: #003b4f !important; /* Dark blue text for readability */
    }
    </style>
""", unsafe_allow_html=True)
# Load trained model
model = joblib.load("model.pkl")
df = pd.read_csv("International_Education_Costs_with_Calculations.csv")  # Replace with your actual dataset file



st.set_page_config(page_title="Study Abroad Cost Estimator", layout="centered")

# Title
st.title("🎓 Study Abroad Cost Estimator")
st.markdown("Fill in your education preferences to estimate the total cost and visualize your input compared to other students.")

# Navigation
menu = st.sidebar.selectbox("Go to", ["🎯 Predict Cost", "📊 Visualize Inputs","📈 EDA Analysis"])

# Predict Page
if menu == "🎯 Predict Cost":

    # Initialize session state
    if "input_data" not in st.session_state:
        st.session_state.input_data = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    st.markdown("### 📝 User Input")
    duration = st.number_input("⏳ Duration (Years)", min_value=0.6, max_value=5.0)
    tuition = st.number_input("💵 Tuition (USD)", min_value=0.0, max_value=62000.0)
    rent = st.number_input("🛏️ Rent (USD)", min_value=150.0, max_value=2500.0)
    visa_fee = st.number_input("🛂 Visa Fee (USD)", min_value=40.0, max_value=490.0)
    insurance = st.number_input("🩺 Insurance (USD)", min_value=200.0, max_value=1500.0)
    exchange_rate = st.number_input("💱 Exchange Rate", min_value=0.0, max_value=42150.0)
    living_cost_index = st.number_input("🏠 Living Cost Index", min_value=27.0, max_value=122.0)

    # Submit Button
    submitted = st.button("🔍 Predict", key="predict_button")

    if submitted:
        input_data = pd.DataFrame({
            "Duration_Years": [duration],
            "Tuition_USD": [tuition],
            "Living_Cost_Index": [living_cost_index],
            "Rent_USD": [rent],
            "Visa_Fee_USD": [visa_fee],
            "Insurance_USD": [insurance],
            "Exchange_Rate": [exchange_rate]
        })

        if input_data.isnull().values.any():
            st.error("❌ Please fill in all fields.")
        else:
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Estimated Total Annual Cost: ${prediction:,.2f}")
            st.session_state.input_data = input_data
            st.session_state.prediction = prediction

# Visualization Page
elif menu == "📊 Visualize Inputs":
    st.title("📊 Visualize Your Inputs")
    
    if st.session_state.get("input_data") is None:
        st.info("📝 Please enter input values in the '🎯 Predict Cost' section first.")
    else:
        input_data = st.session_state["input_data"]

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Duration Graph
        if col1.button("⏳ Show Duration Graph"):
            st.subheader("⏳ Duration (Years)")
            fig, ax = plt.subplots()
            ax.bar(["Duration (Years)"], input_data["Duration_Years"], color=["#2EF9F9"])
            ax.set_ylabel("Years")
            st.pyplot(fig)

                # Tuition Pie Chart
        if col2.button("💵 Show Tuition Fee Graph"):
           st.subheader("💵 Tuition Fee Breakdown")
           tuition_val = input_data["Tuition_USD"][0]
           other_val = max(1, tuition_val * 0.4)
           fig, ax = plt.subplots()
           ax.pie(
           [tuition_val, other_val],
           labels=["Tuition", "Other Costs"],
           colors=["#87CEEB", "#00008B"],  # ✅ Only one '#' per color code
           autopct="%1.1f%%")
    
           st.pyplot(fig)

        # Rent/Visa/Exchange Graph
        if col3.button("📈 Show Rent/Visa/Exchange Graph"):
            st.subheader("📈 Rent, Visa Fee, Exchange Rate")
            values = [
                input_data["Rent_USD"][0],
                input_data["Visa_Fee_USD"][0],
                input_data["Exchange_Rate"][0]
            ]
            labels = ["Rent", "Visa Fee", "Exchange Rate"]
            fig, ax = plt.subplots()
            ax.plot(labels, values, marker='o', linestyle='-', color="#2AF942")
            ax.set_ylabel("Amount (USD)")
            st.pyplot(fig)

        # Living Cost Histogram
        if col4.button("🏠 Show Living Cost Graph"):
            st.subheader("🏠 Living Cost Index")
            fig, ax = plt.subplots()
            ax.hist(input_data["Living_Cost_Index"], bins=5, color="#C214E5", edgecolor='black')
            ax.set_xlabel("Living Cost Index")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
# EDA Section
elif menu == "📈 EDA Analysis":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("🔹 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔸 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Tuition Distribution")
    fig = px.histogram(df, x="Tuition_USD", nbins=30, title="Tuition Fee Distribution", color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig)

    st.subheader("📦 Living Cost by Country")
    fig = px.box(df, x="Country", y="Living_Cost_Index", color="Country", title="Living Cost Comparison")
    st.plotly_chart(fig)

    st.subheader("📉 Tuition vs Total Cost")
    fig = px.scatter(df, x="Tuition_USD", y="Total Annual Cost (USD)", color="Country",
                     size="Rent_USD", title="Tuition vs Total Cost")
    st.plotly_chart(fig)

    with st.expander("🔍 Show Pairplot"):
        fig = sns.pairplot(df.select_dtypes(include='number'))
        st.pyplot(fig)
