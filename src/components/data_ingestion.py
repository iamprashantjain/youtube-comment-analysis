import os
import yaml
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger.logging import logging
from src.exception.exception import customexception

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)["data_ingestion"]
        
        self.test_size = self.params["test_size"]
        self.random_state = self.params["random_state"]

    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Initial data cleaning started")
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        if 'clean_comment' in df.columns:
            df = df[~(df['clean_comment'].str.strip() == "")]
        return df

    def initiate_data_ingestion(self):
        logging.info("Data ingestion process started")
        try:
            # Step 1: Fetch data
            logging.info(f"Fetching data from {self.params['data_url']}")
            df = pd.read_csv(self.params["data_url"])
            
            # Step 2: Clean data
            df = self.initial_data_cleaning(df)
            
            # Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            # Step 3: Train-test split
            logging.info(f"Splitting data (test_size={self.test_size})")
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Step 4: Save outputs
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info(
                f"Data saved to {self.ingestion_config.train_data_path} and "
                f"{self.ingestion_config.test_data_path}"
            )
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.error(f"Failed to run data ingestion: {str(e)}")
        raise