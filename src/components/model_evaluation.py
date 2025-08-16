import os
import sys
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import load_object, save_model_info
import traceback


@dataclass
class ModelEvaluationConfig:
    test_data_path: str = os.path.join("artifacts", "data_preprocessing", "test_processed.csv")
    model_path: str = os.path.join("artifacts", "model_trainer", "lgbm_model.pkl")
    vectorizer_path: str = os.path.join("artifacts", "model_trainer", "tfidf_vectorizer.pkl")

class ModelEvaluation:
    def __init__(self):

        self.config = ModelEvaluationConfig()
        self._setup_mlflow()
        
    def _setup_mlflow(self):

        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)
        
        mlflow_params = self.params["mlflow"]
        
        # Get token from environment variable (required)
        token = os.environ.get("DAGSHUB_PAT")
        
        if not token:
            raise ValueError("DAGSHUB_PAT environment variable not found")
    
        mlflow.set_tracking_uri(
            f"https://{mlflow_params['username']}:{mlflow_params['token']}@dagshub.com/{mlflow_params['repo']}"
        )
        mlflow.set_experiment(mlflow_params["experiment_name"])

    def _log_confusion_matrix(self, cm, class_names):

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        temp_path = os.path.join("artifacts", "confusion_matrix.png")
        plt.savefig(temp_path)
        plt.close()
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)

    def _validate_data(self, df):

        text_column = self.params["data_preprocessing"]["text_column"]
        target_col = self.params["base"]["target_col"]
        
        # Column existence checks
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in test data")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Data cleaning
        df[text_column] = df[text_column].fillna('').astype(str)
        
        if df[text_column].str.strip().eq('').any():
            logging.warning("Found empty strings in text column after cleaning")
        
        return df

    def _log_model_parameters(self, model):

        # Log model-specific parameters if available
        if hasattr(model, 'get_params'):
            for param_name, param_value in model.get_params().items():
                mlflow.log_param(param_name, param_value)
        
        # Log training parameters from config
        mlflow.log_params({
            "model_type": "LightGBM",
            "ngram_range": self.params["model_trainer"]["ngram_range"],
            "max_features": self.params["model_trainer"]["max_features"],
            "learning_rate": self.params["model_trainer"]["learning_rate"],
            "max_depth": self.params["model_trainer"]["max_depth"],
            "n_estimators": self.params["model_trainer"]["n_estimators"]
        })

    def _log_class_metrics(self, report, model):

        if len(model.classes_) <= 10:  # Only log if not too many classes
            for class_name in model.classes_:
                class_name = str(class_name)
                if class_name in report:
                    mlflow.log_metrics({
                        f"precision_{class_name}": report[class_name]["precision"],
                        f"recall_{class_name}": report[class_name]["recall"],
                        f"f1_{class_name}": report[class_name]["f1-score"],
                    })

    def _log_model_artifacts(self, model, X_test, cm, class_names):

        # Log model and vectorizer
        mlflow.sklearn.log_model(model, "lightgbm_model")
        mlflow.log_artifact(self.config.vectorizer_path)
        
        # Log confusion matrix
        self._log_confusion_matrix(cm, class_names)
        
        # Log model with signature
        input_example = X_test[:5]
        signature = mlflow.models.infer_signature(
            input_example, 
            model.predict(input_example)
        )
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
            registered_model_name="youtube-comment-analysis-model"
        )

    def initiate_model_evaluation(self):

        try:
            logging.info("Starting model evaluation")

            # 1. Data Loading and Validation
            test_df = pd.read_csv(self.config.test_data_path)
            test_df = self._validate_data(test_df)
            logging.info(f"Test data loaded. Shape: {test_df.shape}")

            # 2. Model Loading
            model = load_object(self.config.model_path)
            vectorizer = load_object(self.config.vectorizer_path)
            logging.info("Model and vectorizer loaded")            
            
            # 3. Feature Preparation
            text_column = self.params["data_preprocessing"]["text_column"]
            X_test = vectorizer.transform(test_df[text_column])
            y_test = test_df[self.params["base"]["target_col"]].values

            # 4. MLflow Tracking
            with mlflow.start_run() as run:
                logging.info("MLflow run started")

                # 5. Model Prediction and Evaluation
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                # 6. Logging
                self._log_model_parameters(model)
                self._log_class_metrics(report, model)
                self._log_model_artifacts(model, X_test, cm, [str(cls) for cls in model.classes_])
                
                #mlflow model path
                model_path = "lightgbm_model"
                
                # 7. Save model info
                save_model_info(run.info.run_id, model_path, 'experiment_info.json')
                
                logging.info("Model evaluation completed successfully")
                return report

        except Exception as e:
            logging.error(f"Detailed error: {str(e)}")
            logging.error("Full traceback:\n" + traceback.format_exc())
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluation()
        report = evaluator.initiate_model_evaluation()
        print("Evaluation Report:")
        print(pd.DataFrame(report).transpose())
    except Exception as e:
        logging.critical(f"Application failed: {str(e)}")
        sys.exit(1)