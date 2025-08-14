import os
import sys
import yaml
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object

@dataclass
class ModelTrainerConfig:
    processed_train_path: str = os.path.join("artifacts", "data_preprocessing", "train_processed.csv")
    vectorizer_path: str = os.path.join("artifacts", "model_trainer", "tfidf_vectorizer.pkl")
    model_path: str = os.path.join("artifacts", "model_trainer", "lgbm_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)
        
        self.text_column = self.params["data_preprocessing"]["text_column"]
        self.target_column = self.params["base"]["target_col"]
        ngram_list = self.params["model_trainer"]["ngram_range"]
        
        self.tfidf_params = {
            "max_features": self.params["model_trainer"]["max_features"],
            "ngram_range": tuple(ngram_list) if isinstance(ngram_list, list) else (1, 3),
        }
        
        self.lgbm_params = {
            "learning_rate": self.params["model_trainer"]["learning_rate"],
            "max_depth": self.params["model_trainer"]["max_depth"],
            "n_estimators": self.params["model_trainer"]["n_estimators"],
            "objective": "multiclass",
            "random_state": 42
        }

    def _prepare_features(self, data: pd.DataFrame) -> tuple:
        """Apply TF-IDF vectorization and SMOTE oversampling"""
        try:
            logging.info("Starting feature preparation")
            
            # 1. Clean NaN values in text column
            data[self.text_column] = data[self.text_column].fillna("")  # Replace NaN with empty string
            
            # 2. TF-IDF Vectorization
            vectorizer = TfidfVectorizer(**self.tfidf_params)
            X = vectorizer.fit_transform(data[self.text_column])
            y = data[self.target_column].values
            
            logging.info(f"TF-IDF completed. Features shape: {X.shape}")
            
            # 3. Convert sparse matrix to dense for SMOTE
            X_dense = X.toarray()
            
            # 4. SMOTE Oversampling
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_dense, y)
            
            logging.info(f"SMOTE completed. Resampled features shape: {X_res.shape}")
            return X_res, y_res, vectorizer
            
        except Exception as e:
            logging.error(f"Feature preparation failed at step: {str(e)}")
            raise customexception(e, sys)
        

    def initiate_model_training(self):
        try:
            logging.info("Starting model training pipeline")
            
            # 1. Load processed data
            train_df = pd.read_csv(self.config.processed_train_path)
            logging.info(f"Training data loaded: {train_df.shape}")
            
            # 2. Feature engineering
            X_train, y_train, vectorizer = self._prepare_features(train_df)
            
            # 3. LightGBM Training
            model = LGBMClassifier(**self.lgbm_params)
            model.fit(X_train, y_train)
            logging.info("LightGBM training completed")
            
            # 4. Save artifacts
            os.makedirs(os.path.dirname(self.config.vectorizer_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            
            save_object(self.config.vectorizer_path, vectorizer)
            save_object(self.config.model_path, model)
            
            logging.info(
                f"Model artifacts saved:\n"
                f"- Vectorizer: {self.config.vectorizer_path}\n"
                f"- Model: {self.config.model_path}"
            )
            
            return self.config.vectorizer_path, self.config.model_path
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()