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

        # Read params.yaml
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)

        mlflow_params = self.params["mlflow"]

        # Set MLflow configs
        mlflow.set_tracking_uri(f"https://{mlflow_params['username']}:{mlflow_params['token']}@dagshub.com/{mlflow_params['repo']}")
        mlflow.set_experiment(mlflow_params["experiment_name"])

    def _log_confusion_matrix(self, cm, class_names):
        """Log confusion matrix as MLflow artifact"""
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
        """Validate and clean test data"""
        text_column = self.params["data_preprocessing"]["text_column"]
        target_col = self.params["base"]["target_col"]
        
        # Check if required columns exist
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in test data")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Handle missing values
        df[text_column] = df[text_column].fillna('').astype(str)
        
        # Check for empty strings after cleaning
        if df[text_column].str.strip().eq('').any():
            logging.warning("Found empty strings in text column after cleaning")
        
        return df

    def initiate_model_evaluation(self):
        """Execute the model evaluation pipeline"""
        try:
            logging.info("Starting model evaluation")

            # Load and validate test data
            test_df = pd.read_csv(self.config.test_data_path)
            test_df = self._validate_data(test_df)
            logging.info(f"Test data loaded. Shape: {test_df.shape}")

            # Load model and vectorizer
            model = load_object(self.config.model_path)
            vectorizer = load_object(self.config.vectorizer_path)
            logging.info("Model and vectorizer loaded")

            # Prepare features
            text_column = self.params["data_preprocessing"]["text_column"]
            X_test = vectorizer.transform(test_df[text_column])
            y_test = test_df[self.params["base"]["target_col"]].values

            # Start MLflow run
            with mlflow.start_run() as run:
                logging.info("MLflow run started")

                # Log parameters
                mlflow.log_params({
                    "model_type": "LightGBM",
                    "ngram_range": self.params["model_trainer"]["ngram_range"],
                    "max_features": self.params["model_trainer"]["max_features"],
                    "learning_rate": self.params["model_trainer"]["learning_rate"],
                    "max_depth": self.params["model_trainer"]["max_depth"],
                    "n_estimators": self.params["model_trainer"]["n_estimators"]
                })

                # Predict and evaluate
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": report["accuracy"],
                    "weighted_avg_precision": report["weighted avg"]["precision"],
                    "weighted_avg_recall": report["weighted avg"]["recall"],
                    "weighted_avg_f1": report["weighted avg"]["f1-score"]
                })

                
                # Save model info
                model_path = "lgbm_model"
                save_model_info(run.info.run_id, model_path, 'experiment_info.json')
                
                
                # Log class-wise metrics if available
                if len(model.classes_) <= 10:  # Only log if not too many classes
                    for class_name in model.classes_:
                        class_name = str(class_name)
                        if class_name in report:
                            mlflow.log_metrics({
                                f"precision_{class_name}": report[class_name]["precision"],
                                f"recall_{class_name}": report[class_name]["recall"],
                                f"f1_{class_name}": report[class_name]["f1-score"],
                            })

                # Log confusion matrix
                class_names = [str(cls) for cls in model.classes_]
                self._log_confusion_matrix(cm, class_names)

                # Log model with signature
                input_example = X_test[:5]
                signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name="youtube-comment-analysis-model"
                )

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