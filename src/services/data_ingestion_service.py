import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.data_ingestion_config import DataIngestionConfig

logging = LoggerManager.get_logger(__name__)


class DataIngestionService:
    """
    A class for handling the data ingestion process.
    Reads input data, splits it into training and testing datasets, and saves them as CSV files.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class by creating an instance of the DataIngestionConfig.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, test_size=0.2):
        """
        Orchestrates the data ingestion process:
        1. Reads the input data from the specified path.
        2. Splits the data into train and test sets.
        3. Saves the raw, train, and test datasets to their respective paths.

        Args:
            test_size (float): Proportion of the dataset to include in the test split (default is 0.2).

        Returns:
            tuple: Paths to the train and test dataset files.

        Raises:
            CustomException: Custom exception if any error occurs during the process.
        """
        logging.info("Entered the data ingestion method or component.")
        try:
            # Check if the input data file exists
            if not os.path.exists(self.ingestion_config.input_data_path):
                logging.error(
                    f"Input file not found at {self.ingestion_config.input_data_path}"
                )
                raise FileNotFoundError(
                    f"File not found: {self.ingestion_config.input_data_path}"
                )

            # Read the input data into a DataFrame
            df = pd.read_csv(self.ingestion_config.input_data_path)
            logging.info("Read the dataset as a pandas DataFrame.")

            # Ensure the directory for saving artifacts exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved successfully.")

            # Perform train-test split
            logging.info("Initiating train-test split.")
            train_set, test_set = train_test_split(df, test_size=test_size)

            # Save the train and test datasets
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Train and test datasets saved successfully.")

            # Return the file paths of the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            # Log the exception and raise a custom exception
            logging.error("Error occurred during data ingestion: %s", e)
            raise CustomException(e, sys) from e


# if __name__ == "__main__":
#     """
#     Entry point for the script.
#     Initializes the DataIngestion class and executes the data ingestion process.
#     """
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()
