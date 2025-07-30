from fastapi import FastAPI
from pydantic import BaseModel, Field, confloat
import joblib
import numpy as np
import logging
import os
import sqlite3
import datetime
from prometheus_fastapi_instrumentator import Instrumentator

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure file and console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "api.log")),
        logging.StreamHandler()
    ]
)

# Load the trained model
model = joblib.load("models/model.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="Iris Predictor API",
    description="Classifies Iris flower species",
    version="1.0"
)

# Set up Prometheus metrics
Instrumentator().instrument(app).expose(app)


# Input schema with validation
class IrisInput(BaseModel):
    sepal_length: confloat(gt=0, lt=10) = Field(..., example=5.1)
    sepal_width: confloat(gt=0, lt=10) = Field(..., example=3.5)
    petal_length: confloat(gt=0, lt=10) = Field(..., example=1.4)
    petal_width: confloat(gt=0, lt=10) = Field(..., example=0.2)

# Log predictions to SQLite database with timestamp
def log_to_db(input_data: IrisInput, prediction: int):
    db_path = os.path.join(LOG_DIR, 'predictions.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction INTEGER,
            timestamp TEXT
        )
    ''')

    c.execute('''
        INSERT INTO logs (
            sepal_length, sepal_width, petal_length,
            petal_width, prediction, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
        prediction,
        datetime.datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Iris Predictor API!"}

# Prediction endpoint
@app.post("/predict", summary="Predict Iris Species")
def predict(data: IrisInput):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = int(model.predict(X)[0])

    logging.info(f"Input: {data.dict()} → Prediction: {pred}")
    log_to_db(data, pred)

    return {"prediction": pred}