import os
import sys
from datetime import datetime

from sklearn.metrics import accuracy_score

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.services.data_ingestion_service import DataIngestionService
from src.services.data_transformation_service import DataTransformationService
from src.services.model_training_service import ModelTrainingService
from src.services.report_service import ReportService
from src.utils.file_utils import save_json, save_object
from src.utils.history_utils import append_training_history, update_training_history
from src.utils.yaml_loader import load_model_config

logging = LoggerManager.get_logger(__name__)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()
        self.model_training_service = ModelTrainingService()
        self.config = Config()

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion.")
            train_path, validation_path, _, _ = (
                self.data_ingestion_service.initiate_data_ingestion(
                    "Movement_Class"
                )  # Use Movement_Class as the target
            )
            logging.info(
                f"Data ingested. Train: {train_path}, Validation: {validation_path}"
            )

            # Step 2: Data Transformation
            logging.info("Starting data transformation.")
            train_arr, validation_arr, preprocessor_path = (
                self.data_transformation_service.initiate_data_transformation(
                    train_path, validation_path, target_column=self.config.target_column
                )
            )
            logging.info(
                f"Data transformed and preprocessor saved at: {preprocessor_path}"
            )

            # Initialize an empty model_report to capture accuracy scores for all models
            model_report = {}
            model_instances = {}

            # Step 3: Model Training and Selection
            logging.info("Starting model training and selection.")
            model_configs = load_model_config()

            models_to_train = []
            if self.config.best_of_all:
                logging.info("Training all models to find the best.")
                for model_type, _ in model_configs["models"].items():
                    models_to_train.append(model_type)
            elif self.config.model_type:
                model_type = self.config.model_type
                logging.info(f"Training specified models: {model_type}")
                models_to_train = self.config.model_type
                if not models_to_train:
                    raise ValueError(
                        f"Invalid model_type(s) provided: {model_type}. Available models: {self.model_names}"
                    )
            else:
                raise ValueError(
                    "You must specify either --model_type or --best_of_all."
                )

            for model_type in models_to_train:
                train_results = self.model_training_service.train_and_validate(
                    model_type, train_arr, validation_arr
                )
                model_report[model_type] = {
                    "performance_metrics": train_results["performance_metrics"],
                    "best_params": train_results["best_params"],
                    "best_val_accuracy": train_results["best_val_accuracy"],
                }
                model_instances[model_type] = {
                    "model": train_results["model"],
                }
                logging.info(f"Results for {model_type}: {train_results}")

            # Select the best model based on validation accuracy
            best_model_name = max(
                model_report,
                key=lambda m: model_report[m]["performance_metrics"][
                    "validation_accuracy"
                ],
            )
            best_model_results = model_report[best_model_name]
            best_model = model_instances[best_model_name]["model"]

            # Create the final history entry
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": best_model_name,
                "performance_metrics": best_model_results["performance_metrics"],
                "best_params": best_model_results["best_params"],
                "best_val_accuracy": best_model_results["best_val_accuracy"],
                "model_report": model_report,
            }

            # Append to training history file
            update_training_history(history_entry)
            logging.info(f"Training history updated: {history_entry}")

            if self.config.save_best:
                save_object(self.config.MODEL_FILE_PATH, best_model)

            return model_report

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys) from e
