import sys
import os
import yaml
import mlflow
import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
from src.logger.logging import logging
from src.exception.exception import customexception
import traceback

def setup_mlflow():
    """Setup MLflow tracking with Dagshub credentials."""
    with open("params.yaml", "r") as f:
        mlflow_params = yaml.safe_load(f)["mlflow"]
        
    # Get token from environment variable
    token = os.environ.get("DAGSHUB_PAT")
    if not token:
        raise ValueError("DAGSHUB_PAT environment variable not found")

    # Set tracking URI with token
    tracking_uri = f"https://{mlflow_params['username']}:{token}@dagshub.com/{mlflow_params['repo']}"
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


@pytest.mark.parametrize("model_name, stage", [("youtube_chromeplugin_model", "staging"),])
def test_load_latest_staging_model(model_name, stage):
    # Ensure MLflow is configured
    client = setup_mlflow()
    
    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None
    
    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {traceback.format_exc()}")


