import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Inject custom CSS to change sidebar background color and text color
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #002b36 !important; /* Custom dark blue */
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important; /* White text */
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("your_dataset.csv")

# Sidebar navigation
st.sidebar.title("🎓 Study Abroad Cost App")
menu = st.sidebar.radio("Go to", ["🏠 Home", "📈 EDA", "🧮 Predict", "📊 Visualize"])

# 🏠 Home Page
if menu == "🏠 Home":
    st.title("Welcome to the Study Abroad Cost Predictor")
    st.write("This app allows you to explore and predict the cost of studying in different countries.")

# 📈 EDA
elif menu == "📈 EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    st.subheader("📊 Statistical Summary")
    st.write(df.describe())

    st.subheader("🧼 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📉 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Box Plot of Tuition by Country")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="Country", y="Tuition_USD", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("🔵 Scatter Plot: Tuition vs. Total Cost")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Tuition_USD", y="Total Annual Cost (USD)", data=df, ax=ax)
    st.pyplot(fig)

# 🧮 Predict
elif menu == "🧮 Predict":
    st.title("Predict Total Annual Cost")

    country = st.selectbox("Select Country", df["Country"].unique())
    level = st.selectbox("Select Level", df["Level"].unique())
    duration = st.number_input("Duration (years)", min_value=1)
    tuition = st.number_input("Tuition (USD)", min_value=0.0)
    living_cost = st.number_input("Living Cost Index", min_value=0.0)
    rent = st.number_input("Monthly Rent (USD)", min_value=0.0)
    visa = st.number_input("Visa Fee (USD)", min_value=0.0)
    insurance = st.number_input("Insurance (USD)", min_value=0.0)
    exchange_rate = st.number_input("Exchange Rate", min_value=0.0)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "Country": [country],
            "Level": [level],
            "Duration_Years": [duration],
            "Tuition_USD": [tuition],
            "Living_Cost_Index": [living_cost],
            "Rent_USD": [rent],
            "Visa_Fee_USD": [visa],
            "Insurance_USD": [insurance],
            "Exchange_Rate": [exchange_rate]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Total Annual Cost: ${prediction:,.2f}")

# 📊 Visualize (placeholder)
elif menu == "📊 Visualize":
    st.title("Visualize Results")
    st.write("Add interactive charts or visual summaries here.")
