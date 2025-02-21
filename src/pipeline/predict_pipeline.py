import os
import sys
from collections import namedtuple

import pandas as pd

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.model_container import ModelContainer
from src.services.data_ingestion_service import DataIngestionService
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
            self.ingestion_service = DataIngestionService()

            logging.info("Loading model and preprocessor.")

            # Load the model and preprocessor once during initialization
            model_container = ModelContainer.load(model_path)
            self.model = model_container.model
            self.transformer = model_container.transformer

            logging.info("Model and transformer loaded successfully.")
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

            logging.info("Computing necessary features.")
            preprocessed_features = self.ingestion_service.preprocess_data(features)

            # Preprocess the features
            data_scaled = self.transformer.transform(preprocessed_features)
            logging.info("Data transformed successfully.")

            # Make predictions
            y_hat = self.model.predict(data_scaled)
            logging.info("Prediction completed successfully.")

            y_hat = y_hat.astype(int)
            y_hat = y_hat.flatten()

            y_proba = self.model.predict_proba(data_scaled)
            y_confidence = y_proba.max(axis=1)
            PredictionResult = namedtuple(
                "PredictionResult", ["predictions", "confidence"]
            )
            return PredictionResult(y_hat, y_confidence)

        except Exception as e:
            raise CustomException(e, sys) from e
