from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import train_binary_classifier, predict
from app.counterfactuals import generate_counterfactuals
from app.preprocess import preprocess_data
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the CTR Prediction API!"}

@app.post("/train/")
async def train(file: UploadFile = File(...)):
    try:
        # Load and preprocess data
        data = pd.read_csv(file.file)
        preprocessed_data = preprocess_data(data)

        # Train the model
        train_binary_classifier(preprocessed_data)
        return {"message": "Model trained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        logging.info(f"Input data shape: {data.shape}")
        preprocessed_data = preprocess_data(data)
        logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        predictions = predict(preprocessed_data)
        logging.info(f"Predictions length: {len(predictions)}")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


@app.post("/counterfactual/")
async def counterfactual_endpoint(file: UploadFile = File(...), user_index: int = 0):
    try:
        # Load and preprocess data
        data = pd.read_csv(file.file)
        preprocessed_data = preprocess_data(data)

        # Combine features and target for training data
        full_data = pd.concat([preprocessed_data.drop(columns=['Clicked on Ad']), preprocessed_data['Clicked on Ad']], axis=1)

        # Select user data for counterfactual generation
        user_data = preprocessed_data.iloc[[user_index]].drop(columns=['Clicked on Ad'])

        # Generate counterfactuals
        counterfactuals = generate_counterfactuals(full_data, user_data, desired_class=1)
        return {"counterfactuals": counterfactuals.to_json(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during counterfactual generation: {str(e)}")
