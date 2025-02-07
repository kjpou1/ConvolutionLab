import os
import sys

import pandas as pd

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.utils.file_utils import load_object

logging = LoggerManager.get_logger(__name__)


class PredictPipeline:
    def __init__(self):
        """
        Initialize the PredictPipeline by loading the model and preprocessor.
        """
        try:
            self.config = Config()
            model_path = self.config.MODEL_FILE_PATH
            preprocessor_path = self.config.PREPROCESSOR_FILE_PATH
            logging.info("Loading model and preprocessor.")

            # Load the model and preprocessor once during initialization
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Model and preprocessor loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, features):
        """
        Predict outcomes based on the given features.

        Args:
            features (pd.DataFrame or np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        try:
            logging.info("Starting prediction.")

            # Preprocess the features
            data_scaled = self.preprocessor.transform(features)
            logging.info("Data transformed successfully.")

            # Make predictions
            preds = self.model.predict(data_scaled)
            logging.info("Prediction completed successfully.")

            return preds
        except Exception as e:
            raise CustomException(e, sys) from e
