import mlflow.sklearn
import joblib
import os

# Set tracking URI (local mlruns folder)
mlflow.set_tracking_uri("file:./mlruns")

# Load the sklearn model directly from the registry
model = mlflow.sklearn.load_model("models:/IrisClassifier/Production")

# Save as .pkl
joblib.dump(model, "models/model.pkl")

print(" Model saved as models/model.pkl")
