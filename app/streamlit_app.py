import streamlit as st
import pandas as pd
import joblib
from app.preprocess import preprocess_data
from app.counterfactuals import generate_counterfactuals

# Load the pre-trained model
MODEL_PATH = "app/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

# App Title
st.title("Ad Click Prediction and Counterfactual Explanations")

# Upload Dataset
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read and preprocess data
    data = pd.read_csv(uploaded_file)
    st.write("Original Dataset:")
    st.dataframe(data.head())

    # Preprocess the data
    processed_data = preprocess_data(data)
    st.write("Processed Dataset:")
    st.dataframe(processed_data.head())

    # Predict using the model
    if st.button("Predict"):
        predictions = model.predict(processed_data)
        processed_data["Prediction"] = predictions
        st.write("Prediction Results:")
        st.dataframe(processed_data[["Prediction"]].head())

    # Generate Counterfactuals
    st.header("Generate Counterfactual Explanations")
    user_index = st.number_input("Select Row Index for Counterfactual Explanation", min_value=0, max_value=len(processed_data)-1, step=1)

    if st.button("Generate Counterfactuals"):
        user_data = processed_data.iloc[[user_index]].copy()
        full_data = pd.concat([processed_data, data['Clicked on Ad']], axis=1)  # Include target for DiCE
        cf = generate_counterfactuals(model, full_data, user_data, desired_class=1)
        st.write("Counterfactual Explanations:")
        st.dataframe(cf)
