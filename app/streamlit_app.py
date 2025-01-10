import streamlit as st
import requests
import pandas as pd

# Define the API URL
API_URL = "http://localhost:8000"

st.title("CTR Prediction Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.write(df.head())

        # Train Model
        if st.button("Train Model"):
            uploaded_file.seek(0)  # Reset file pointer
            response = requests.post(
                f"{API_URL}/train/",
                files={"file": ("uploaded_file.csv", uploaded_file.read(), "text/csv")}
            )
            st.write(response.json())

        if st.button("Make Predictions"):
            uploaded_file.seek(0)  # Reset file pointer
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": ("uploaded_file.csv", uploaded_file.read(), "text/csv")}
            )

            # Extract predictions and handle errors
            try:
                predictions = response.json().get("predictions", [])
                if len(predictions) != len(df):
                    st.error("Prediction failed: Mismatch in data length.")
                else:
                    df["Predictions"] = predictions
                    st.write("Predictions Added to Dataset:", df)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        if st.button("Generate Counterfactuals"):
            user_index = st.number_input("Enter User Index for Counterfactuals", min_value=0, max_value=len(df) - 1, step=1)
            uploaded_file.seek(0)  # Reset file pointer
            response = requests.post(
                f"{API_URL}/counterfactual/",
                files={"file": ("uploaded_file.csv", uploaded_file.read(), "text/csv")},
                data={"user_index": user_index},
            )
            st.write("Counterfactuals:", response.json())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
