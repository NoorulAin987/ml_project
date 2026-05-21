import streamlit as st
from components import *
from pages import *
from utils import *

def main():
    st.title("Credit Card Fraud Detection")
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox("Select a Page", ["Home", "Data Exploration", "Model Prediction"])
    
    if page == "Home":
        show_home()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Model Prediction":
        show_model_prediction()

if __name__ == "__main__":
    main()