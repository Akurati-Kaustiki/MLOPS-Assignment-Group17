from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import mlflow.pyfunc
import os

# Load model
# mlflow.set_tracking_uri("file:./mlruns")
# model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")
model = joblib.load("models/model.pkl")


app = FastAPI()
logging.basicConfig(filename='logs/api.log', level=logging.INFO)

class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length in cm (0-10)")
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width in cm (0-10)")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length in cm (0-10)")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width in cm (0-10)")

@app.post("/predict")
def predict(data: IrisInput):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = int(model.predict(X)[0])
    logging.info(f"Input: {data.dict()} -> Prediction: {pred}")
    return {"prediction": pred}