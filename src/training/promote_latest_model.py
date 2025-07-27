from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

client = MlflowClient()

model_name = "IrisClassifier"

# Find latest version
latest_versions = client.get_latest_versions(name=model_name)
latest_version = max(int(v.version) for v in latest_versions)

# Promote latest to Production
client.transition_model_version_stage(
    name=model_name,
    version=str(latest_version),
    stage="Production",
    archive_existing_versions=True
)

print(f"Promoted IrisClassifier version {latest_version} to Production")
