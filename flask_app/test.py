from src.utils.utils import *
import yaml
import mlflow
from mlflow.tracking import MlflowClient
import os
from pathlib import Path
from src.logger.logging import logging
from src.exception.exception import customexception
import traceback
import sys

try:
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    logging.info("Successfully loaded params.yaml")

    # Configure MLflow tracking
    mlflow_params = params["mlflow"]
    mlflow.set_tracking_uri(f"https://{mlflow_params['username']}:{mlflow_params['token']}@dagshub.com/{mlflow_params['repo']}")
    logging.info(f"Configured MLflow tracking URI for {mlflow_params['repo']}")

    def load_model_from_registry(model_name: str, version: str):
        """Load a model from MLflow model registry"""
        try:
            logging.info(f"Attempting to load model {model_name} version {version}")
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"Successfully loaded model {model_name} version {version}")
            return model
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            logging.error(traceback.format_exc())
            raise customexception(e, sys)

    # Example usage
    try:
        model = load_model_from_registry("youtube_chromeplugin_model", "2")
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        logging.error(traceback.format_exc())
        raise customexception(e, sys)

except Exception as e:
    logging.critical(f"Application failed: {str(e)}")
    logging.critical(traceback.format_exc())
    raise customexception(e, sys)
