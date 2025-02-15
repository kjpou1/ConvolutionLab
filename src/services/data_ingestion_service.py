import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import src.indicators.leavitt_indicator as lu
from src.exception import CustomException
from src.indicators.feature_engineering import compute_indicators, compute_leavitt_data
from src.indicators.movement import classify_movement
from src.logger_manager import LoggerManager
from src.models.data_ingestion_config import DataIngestionConfig
from src.models.data_processing_error import DataProcessingError
from src.services.data_split_service import DataSplitService

logging = LoggerManager.get_logger(__name__)


class DataIngestionService:
    """
    A class for handling the data ingestion process.
    Reads input data, applies initial preprocessing, identifies features, and splits it into train/test datasets.
    """

    def __init__(self):
        """
        Initializes the DataIngestionService with the configuration.
        """
        self.ingestion_config = DataIngestionConfig()
        self.data_split_service = DataSplitService()

    def preprocess_data(self, df: pd.DataFrame):
        """
        Performs initial data cleaning, column renaming, and drops unnecessary columns.

        Args:
            df (pd.DataFrame): The raw dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        try:
            # Convert 'time' column to datetime and rename to 'Date'
            if "time" in df.columns:
                df = df.rename(columns={"time": "Date"}).assign(
                    Date=lambda x: pd.to_datetime(x["Date"])
                )
                logging.info("Converted 'time' column to 'Date' with datetime format.")

            # Drop unnecessary bid and ask price columns
            columns_to_drop = [
                "bid_o",
                "bid_h",
                "bid_l",
                "bid_c",
                "ask_o",
                "ask_h",
                "ask_l",
                "ask_c",
            ]
            df.drop(
                columns=[col for col in columns_to_drop if col in df.columns],
                errors="ignore",
                inplace=True,
            )
            logging.info("Dropped bid and ask price columns.")

            # Rename mid-price columns
            rename_mapping = {
                "mid_o": "Open",
                "mid_c": "Close",
                "mid_h": "High",
                "mid_l": "Low",
                "volume": "Volume",
            }
            df = df.rename(
                columns={k: v for k, v in rename_mapping.items() if k in df.columns}
            )
            logging.info("Renamed mid-price columns to OHLC format.")

            # Drop 'Unnamed: 0' index column if present
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
                logging.info("Dropped 'Unnamed: 0' index column.")

            # Drop rows where 'Date' is NaN before setting index
            df = df.dropna(subset=["Date"])
            df = df.set_index("Date")

            # Remove duplicate timestamps to ensure unique index
            df = df[~df.index.duplicated(keep="first")]
            logging.info("Set index to Date with unique timestamps.")

            logging.info("Processing leavitt data.")
            df = compute_leavitt_data(df)
            logging.info("Finished leavitt data.")

            logging.info("Processing indicators.")
            df = compute_indicators(df)
            logging.info("Finished indicators.")

            logging.info("Processing target.")
            df = classify_movement(df)
            logging.info("Finished target.")

            # **Drop NaNs from all columns except future target fields ("Target_T+*")**
            columns_to_clean = [
                col for col in df.columns if not col.startswith("Target_T+")
            ]
            df.dropna(subset=columns_to_clean, inplace=True)
            logging.info(f"Dropped NaNs from all columns except Target_T+* fields.")

            if df.isnull().all().sum() > 0:
                raise DataProcessingError("All rows contain NaNs after processing!")

            logging.info(f"[Data Preprocessing] Input Data Shape: {df.shape}")
            logging.info(
                f"[Data Preprocessing] Checking for NaNs: {df.isnull().sum().sum()} NaNs found"
            )

            return df

        except DataProcessingError as e:
            logging.error(f"Data Processing Error: {e}")
            raise CustomException(e, sys) from e

        except Exception as e:
            raise CustomException(e, sys) from e

    def identify_features(self, df: pd.DataFrame, target_column: str = None):
        """
        Identifies numerical, categorical, and target features in the dataset.

        Args:
            df (pd.DataFrame): The input dataset.
            target_column (str): The name of the target column.

        Returns:
            dict: A dictionary containing numerical, categorical, and target features.
        """
        try:
            numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

            # Remove the target column from features
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            if target_column in categorical_features:
                categorical_features.remove(target_column)

            # 🔹 Ensure ALL Leavitt features are captured:
            expected_features = [
                "Leavitt_Projection",
                "Leavitt_Convolution",
                "LC_Slope",
                "LC_Acceleration",
                "Convolution_Probability",
                "Momentum_T-1",
                "Momentum_T-2",
                "Momentum_T-5",
                "Momentum_T-10",
                "Momentum_T-21",
                "Returns_T-1",
                "Returns_T-2",
                "Returns_T-5",
                "Returns_T-10",
                "Returns_T-21",
                "Hour",
                "Day_Of_Week",
                "Month",
                "Year",
                "AHMA",
                "ATR",
            ]
            for feature in expected_features:
                if feature not in numerical_features + categorical_features:
                    logging.warning(f"Feature {feature} is missing from dataset!")

            logging.info("Numerical features identified: %s", numerical_features)
            logging.info("Categorical features identified: %s", categorical_features)
            logging.info("Target column: %s", target_column)

            return {
                "numerical": numerical_features,
                "categorical": categorical_features,
                "target": target_column,
            }
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self, target_column: str, val_size=0.2, test_size=0.2):
        """
        Orchestrates the data ingestion process:
        - Reads the input data.
        - Applies preprocessing (renaming columns, dropping unnecessary fields).
        - Identifies features.
        - Splits the data into train, validation, and test sets.
        - Saves the raw, train, validation, and test datasets.

        Args:
            target_column (str): The name of the target column.
            val_size (float): Proportion of the train dataset to include in the validation split.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Paths to the train, validation, and test dataset files, and identified features.

        Raises:
            CustomException: Custom exception if any error occurs during the process.
        """
        logging.info("Entered the data ingestion method.")
        try:
            if not os.path.exists(self.ingestion_config.input_data_path):
                logging.error(
                    f"Input file not found at {self.ingestion_config.input_data_path}"
                )
                raise FileNotFoundError(
                    f"File not found: {self.ingestion_config.input_data_path}"
                )

            df = pd.read_csv(self.ingestion_config.input_data_path)
            logging.info("Read the dataset as a pandas DataFrame.")

            # Apply preprocessing (renaming, dropping unnecessary columns)
            df_cleaned = self.preprocess_data(df)

            # Identify features
            feature_metadata = self.identify_features(df_cleaned, target_column)

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df_cleaned.to_csv(
                self.ingestion_config.raw_data_path, index=False, header=True
            )
            logging.info("Raw dataset saved successfully.")

            train_set, val_set, test_set = self.data_split_service.sequential_split(
                df_cleaned, test_size=test_size, val_size=val_size
            )

            # Save the datasets
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            val_set.to_csv(
                self.ingestion_config.validation_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Train, validation, and test datasets saved successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path,
                feature_metadata,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
