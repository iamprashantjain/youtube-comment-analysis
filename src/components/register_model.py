import sys
import os
import yaml
import os
import json
import mlflow
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import customexception
import traceback

@dataclass
class ModelRegistrationConfig:
    model_info_path: str = 'experiment_info.json'
    model_name: str = "youtube_chromeplugin_model"

class ModelRegistration:
    def __init__(self):
        self.config = ModelRegistrationConfig()
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        with open("params.yaml", "r") as f:
            mlflow_params = yaml.safe_load(f)["mlflow"]
        
        # Get token from environment variable
        token = os.environ.get("DAGSHUB_PAT")
        if not token:
            raise ValueError("DAGSHUB_PAT environment variable not found")

        # Use environment token instead of mlflow_params['token']
        mlflow.set_tracking_uri(
            f"https://{mlflow_params['username']}:{token}@dagshub.com/{mlflow_params['repo']}"
        )


    def _load_model_info(self) -> dict:
        try:
            with open(self.config.model_info_path, 'r') as file:
                model_info = json.load(file)
            logging.info('Model info loaded successfully')
            return model_info
        except FileNotFoundError:
            error_msg = f'File not found: {self.config.model_info_path}'
            logging.error(error_msg)
            raise customexception(error_msg, sys)
        except Exception as e:
            error_msg = f'Error loading model info: {str(e)}'
            logging.error(error_msg)
            raise customexception(error_msg, sys)

    def _register_model_version(self, model_info: dict):
        """Register a new version of the model"""
        try:
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            return mlflow.register_model(model_uri, self.config.model_name)
        except Exception as e:
            error_msg = f'Model registration failed: {str(e)}'
            logging.error(error_msg)
            raise customexception(error_msg, sys)

    def _transition_to_stage(self, version, stage: str = "Staging"):
        """Transition model version to specified stage bcoz when we register model it will be in staging"""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.config.model_name,
                version=version,
                stage=stage
            )
            logging.info(f'Model transitioned to {stage} stage')
        except Exception as e:
            error_msg = f'Stage transition failed: {str(e)}'
            logging.error(error_msg)
            raise customexception(error_msg, sys)

    
    def initiate_model_registration(self):
        """Execute the model registration pipeline"""
        try:
            logging.info("Starting model registration")

            # 1. Load model information
            model_info = self._load_model_info()
            
            # 2. Register new model version
            model_version = self._register_model_version(model_info)
            
            # 3. Transition to staging
            self._transition_to_stage(model_version.version)
            
            logging.info("Model registration completed successfully")
            return {
                "model_name": self.config.model_name,
                "version": model_version.version,
                "stage": "Staging"
            }

        except Exception as e:
            logging.error(f"Detailed error: {str(e)}")
            logging.error("Full traceback:\n" + traceback.format_exc())
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        registrar = ModelRegistration()
        result = registrar.initiate_model_registration()
        print("Registration Result:")
        print(result)
    except Exception as e:
        logging.critical(f"Application failed: {str(e)}")
        sys.exit(1)