import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure matplotlib
import matplotlib
matplotlib.use("Agg")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "credit_card_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the correct directory.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# App title
st.title("Credit Card Fraud Detection")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an option:", ["Home", "Upload Dataset", "Predict Fraud"])

# Home Page
if options == "Home":
    st.header("Welcome to the Credit Card Fraud Detection App")
    st.write("""
    This app allows you to:
    - Upload a dataset and visualize it.
    - Predict whether a transaction is fraudulent or not.
    - Use a trained Random Forest model for predictions.
    """)

# Upload Dataset Page
elif options == "Upload Dataset":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Visualize class distribution
        if "Class" in df.columns:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Class", data=df, palette="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("The dataset does not contain a 'Class' column.")

# Predict Fraud Page
elif options == "Predict Fraud":
    st.header("Predict Fraud for a Single Transaction")

    # Input fields for transaction features
    st.subheader("Enter Transaction Details:")
    v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
    amount = st.number_input("Amount", value=0.0)
    time = st.number_input("Time", value=0.0)

    # Predict button
    if st.button("Predict"):
        input_data = np.array([v_features + [amount, time]])
        prediction = model.predict(input_data)[0]
        if prediction == 0:
            st.success("The transaction is NOT fraudulent.")
        else:
            st.error("The transaction is FRAUDULENT.")

# Fallback for invalid options
else:
    st.error("Invalid option selected. Please choose a valid option from the sidebar.")