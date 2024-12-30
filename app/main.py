from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from app.preprocess import preprocess_data
from app.model import train_model
from app.counterfactuals import generate_counterfactuals

app = FastAPI()

# Load Model
MODEL_PATH = "app/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

@app.post("/train/")
async def train(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    data = preprocess_data(data)
    train_model(data, MODEL_PATH)
    return {"message": "Model trained and saved successfully!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    predictions = model.predict(data)
    return {"predictions": predictions.tolist()}

@app.post("/counterfactual/")
async def counterfactual(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    cf = generate_counterfactuals(model, data, data.iloc[:1])
    return cf.to_dict()
