import os
import sys
import re
import yaml
import pandas as pd
from dataclasses import dataclass
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object

@dataclass
class DataPreprocessingConfig:
    raw_train_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    raw_test_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    processed_train_path: str = os.path.join("artifacts", "data_preprocessing", "train_processed.csv")
    processed_test_path: str = os.path.join("artifacts", "data_preprocessing", "test_processed.csv")


class DataPreprocessing:
    def __init__(self):
        self.config = DataPreprocessingConfig()
        
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english')) - {'not', 'no', 'but', 'however'}     #keeping these stopwords as it will help in analyzing sentiments
        self.lemmatizer = WordNetLemmatizer()
        
        # Get config from params.yaml
        self.text_column = self.params["data_preprocessing"]["text_column"]

    def _preprocess_text(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                return ""
            
            # Lowercase
            text = text.lower().strip()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special chars (keep basic punctuation)
            text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
            
            # Remove newlines
            text = re.sub(r'\n', ' ', text)
            
            # Tokenize and process words
            words = word_tokenize(text)
            words = [
                self.lemmatizer.lemmatize(word) 
                for word in words 
                if word not in self.stop_words and len(word) > 2
            ]
            
            return ' '.join(words)
        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            return text  # Return original if error occurs

    def initiate_preprocessing(self):
        try:
            os.makedirs(os.path.dirname(self.config.processed_train_path), exist_ok=True)
            
            # Read raw data
            train_df = pd.read_csv(self.config.raw_train_path)
            test_df = pd.read_csv(self.config.raw_test_path)
            
            logging.info(f"Raw data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Apply preprocessing
            train_df[self.text_column] = train_df[self.text_column].apply(self._preprocess_text)
            test_df[self.text_column] = test_df[self.text_column].apply(self._preprocess_text)
            
            # Save processed data
            train_df.to_csv(self.config.processed_train_path, index=False)
            test_df.to_csv(self.config.processed_test_path, index=False)
            
            logging.info(f"Processed data saved to:\n"
                       f"- Train: {self.config.processed_train_path}\n"
                       f"- Test: {self.config.processed_test_path}")
            
            return (
                self.config.processed_train_path,
                self.config.processed_test_path
            )
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    preprocessor = DataPreprocessing()
    train_path, test_path = preprocessor.initiate_preprocessing()