import yaml
import os
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow():
    """Setup MLflow tracking with Dagshub credentials."""
    with open("params.yaml", "r") as f:
        mlflow_params = yaml.safe_load(f)["mlflow"]

    token = os.environ.get("DAGSHUB_PAT")
    if not token:
        raise ValueError("DAGSHUB_PAT environment variable not found")

    tracking_uri = (
        f"https://{mlflow_params['username']}:{token}"
        f"@dagshub.com/{mlflow_params['repo']}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def promote_model():
    client = setup_mlflow()

    model_name = "youtube_chromeplugin_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()