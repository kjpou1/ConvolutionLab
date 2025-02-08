import os
from dataclasses import dataclass

from src.config.config import Config


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    Defines the file paths and directories for data ingestion.
    """

    config: Config = Config()  # Access the centralized Config singleton

    # File paths
    input_data_path: str = config.input_data_file_path
    train_data_path: str = os.path.join(
        config.PROCESSED_DATA_DIR, "train.csv"
    )  # Training data
    validation_data_path: str = os.path.join(
        config.PROCESSED_DATA_DIR, "validation.csv"
    )  # Testing data
    test_data_path: str = os.path.join(
        config.PROCESSED_DATA_DIR, "test.csv"
    )  # Testing data
    raw_data_path: str = os.path.join(config.RAW_DATA_DIR, "data.csv")  # Raw dataset
