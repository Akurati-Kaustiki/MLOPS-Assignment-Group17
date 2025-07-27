import mlflow
import joblib
import os

# Set tracking URI (local mlruns folder)
mlflow.set_tracking_uri("file:./mlruns")

# Load the model from registry (Production version)
model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")

# Extract the actual model object (if it's sklearn, keras, etc.)
native_model = model._model_impl.python_model.model

# Save it as .pkl
os.makedirs("models", exist_ok=True)
joblib.dump(native_model, "models/model.pkl")

print("✅ Model saved as models/model.pkl")
