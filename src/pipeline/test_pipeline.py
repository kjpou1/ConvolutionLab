import os
import sys
from datetime import datetime

import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.model_container import ModelContainer
from src.services.data_ingestion_service import DataIngestionService
from src.services.data_transformation_service import DataTransformationService

# from src.services.model_training_service import ModelTestingService
from src.services.model_testing_service import ModelTestingService
from src.services.report_service import ReportService
from src.utils.file_utils import save_json, save_json_safe, save_object
from src.utils.history_utils import append_training_history, update_training_history
from src.utils.yaml_loader import load_model_config

logging = LoggerManager.get_logger(__name__)


class TestPipeline:
    def __init__(self):
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()

        self.config = Config()
        model_path = self.config.MODEL_FILE_PATH

        logging.info("Loading model and preprocessor.")
        # Load the model and preprocessor once during initialization
        model_container = ModelContainer.load(model_path)
        self.model = model_container.model
        self.transformer = model_container.transformer

        self.model_testing_service = ModelTestingService(self.model)

        logging.info("Model and transformer loaded successfully.")

    def run_pipeline(self):
        try:
            logging.info("Starting test pipeline.")

            # preprocessed_features = self.ingestion_service.preprocess_data(features)
            # Step 1: Data Ingestion
            logging.info("Loading test dataset.")
            train_path, validation_path, test_path, _ = (
                self.data_ingestion_service.initiate_data_ingestion("Movement_Class")
            )  # Test set is included here
            logging.info(f"Test data path: {test_path}")

            # Step 2: Data Transformation
            test_arr = (
                self.data_transformation_service.initiate_data_transformation_for_test(
                    test_path=test_path,
                    target_column=self.config.target_column,
                    preprocessing_obj=self.transformer,
                )
            )
            logging.info(
                f"Test data transformation complete.  Test Featurs Shape: {test_arr.shape}"
            )

            # Step 4: Run Predictions
            y_test, y_pred = self.model_testing_service.run_predictions(test_arr)

            # Step 5: Calculate Metrics
            logging.info("Calculating test set performance metrics.")

            ## Classification Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred)

            ## Regression Metrics (for price forecasting)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Prevent division by zero in MAPE
            nonzero_mask = y_test != 0
            mape = (
                np.mean(
                    np.abs(
                        (y_test[nonzero_mask] - y_pred[nonzero_mask])
                        / y_test[nonzero_mask]
                    )
                )
                * 100
            )

            # Prevent division by zero in SMAPE
            denominator = np.abs(y_test) + np.abs(y_pred)
            denominator_mask = denominator != 0  # Avoid divide-by-zero
            smape = (
                np.mean(
                    2
                    * np.abs(y_test[denominator_mask] - y_pred[denominator_mask])
                    / denominator[denominator_mask]
                )
                * 100
            )

            ## Directional Accuracy
            directional_accuracy = np.mean(
                np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_test[1:] - y_test[:-1])
            )

            ## Error Distribution Statistics
            errors = y_test - y_pred
            error_distribution = {
                "Mean Error": np.mean(errors),
                "Median Error": np.median(errors),
                "Standard Deviation": np.std(errors),
                "Skewness": stats.skew(errors),
                "Kurtosis": stats.kurtosis(errors),
            }

            # Step 6: Save Test Results
            model_report = {
                "Test Results": {
                    "Test Set Accuracy": accuracy,
                    "Test Set MAE": mae,
                    "Test Set RMSE": rmse,
                    "Test Set MAPE": mape,
                    "Test Set SMAPE": smape,
                    "Test Set F1 Macro": f1_macro,
                    "Test Set F1 Weighted": f1_weighted,
                    "Test Set Directional Accuracy": directional_accuracy,
                    "Test Set Confusion Matrix": conf_matrix,
                    "Test Set Classification Report": class_report,
                    "Test Set Error Distribution": error_distribution,
                }
            }

            results_path = os.path.join(self.config.REPORTS_DIR, "test_results.json")
            save_json_safe(model_report, results_path)
            logging.info(f"Test results saved to {results_path}")

            return model_report

        except Exception as e:
            logging.error(f"Error in test pipeline: {e}")
            raise CustomException(e, sys) from e
