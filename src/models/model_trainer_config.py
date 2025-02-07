import os
from dataclasses import dataclass

from src.config.config import Config


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for ModelTrainer.
    Defines paths and directories used during model training and evaluation.
    """

    config = Config()
    trained_model_file_path: str = config.MODEL_FILE_PATH
    catboost_training_dir: str = os.path.join(config.LOG_DIR, "catboost_logs")
