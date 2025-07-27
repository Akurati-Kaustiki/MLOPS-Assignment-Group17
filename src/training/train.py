# src/models/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target", "target_name"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Set tracking and experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_Classification")

best_acc = 0
best_model = None
best_model_name = ""
best_run_id = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Track the best model
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id

# Save best model locally
os.makedirs("models", exist_ok=True)
model_path = f"models/{best_model_name}.pkl"
joblib.dump(best_model, model_path)
print(f"Saved best model locally as {model_path}")

# Register best model in MLflow Model Registry
model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "IrisClassifier"

mlflow.register_model(model_uri=model_uri, name=registered_model_name)
print(f"Registered best model ({best_model_name}) with accuracy {best_acc} to MLflow Registry as '{registered_model_name}'")