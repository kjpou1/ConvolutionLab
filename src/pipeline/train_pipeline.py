import os
import sys
from datetime import datetime

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.services.data_ingestion_service import DataIngestionService
from src.services.data_transformation_service import DataTransformationService
from src.services.model_selection_service import ModelSelectionService
from src.services.model_training_service import ModelTrainingService
from src.utils.file_utils import save_json, save_object
from src.utils.history_utils import append_training_history, update_training_history
from src.utils.yaml_loader import load_model_config

logging = LoggerManager.get_logger(__name__)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()
        self.model_selection_service = ModelSelectionService()
        self.model_training_service = ModelTrainingService()
        self.config = Config()

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion.")
            train_path, test_path = (
                self.data_ingestion_service.initiate_data_ingestion()
            )
            logging.info(f"Data ingested. Train: {train_path}, Test: {test_path}")

            # Step 2: Data Transformation
            logging.info("Starting data transformation.")
            train_arr, test_arr, preprocessor_path = (
                self.data_transformation_service.initiate_data_transformation(
                    train_path, test_path
                )
            )
            logging.info(
                f"Data transformed and preprocessor saved at: {preprocessor_path}"
            )

            # Initialize an empty model_report to capture scores for all models
            model_report = {}
            model_instances = {}

            # Step 3: Model Training and Selection
            logging.info("Starting model training and selection.")
            model_configs = load_model_config()
            # Initialize models and their parameters
            models = {}
            params = {}

            models_to_train = []
            # Determine models to train
            if self.config.best_of_all:
                logging.info("Training all models to find the best.")
                for model_type, _ in model_configs["models"].items():
                    models_to_train.append(model_type)
            elif self.config.model_type:
                model_type = self.config.model_type
                logging.info(f"Training specified models: {model_type}")
                # models_to_train = [
                #     model for model in model_type if model in self.model_names
                # ]
                models_to_train = self.config.model_type
                if not models_to_train:
                    raise ValueError(
                        f"Invalid model_type(s) provided: {model_type}. "
                        f"Available models: {self.model_names}"
                    )
            else:
                raise ValueError(
                    "You must specify either --model_type or --best_of_all."
                )

            # for model_type, model_info in model_configs["models"].items():
            for model_type in models_to_train:
                train_results = self.model_training_service.train_and_validate(
                    model_type, train_arr, test_arr
                )
                model_report[model_type] = {
                    "train_r2": train_results["train_r2"],
                    "test_r2": train_results["test_r2"],
                }
                model_instances[model_type] = {
                    "model": train_results["model"],
                }
                logging.info(f"Results for {model_type}: {train_results}")

            # Once all models are trained, select the best one based on test_r2
            best_model_name = max(
                model_report, key=lambda m: model_report[m]["test_r2"]
            )
            best_model_results = model_report[best_model_name]
            best_model = model_instances[best_model_name]["model"]

            # Create the final history entry
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": best_model_name,
                "train_r2": best_model_results["train_r2"],
                "test_r2": best_model_results["test_r2"],
                "model_report": model_report,
            }

            # Append to the centralized training history file
            update_training_history(history_entry)
            logging.info(f"Training history updated: {history_entry}")

            if self.config.save_best:
                save_object(self.config.MODEL_FILE_PATH, best_model)

            return model_report

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys) from e
