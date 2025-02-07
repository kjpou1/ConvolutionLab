import logging
import os
import uuid
from datetime import datetime

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config.config import Config
from src.exception import CustomException
from src.models.model_trainer_config import ModelTrainerConfig
from src.utils.file_utils import save_training_artifacts
from src.utils.ml_utils import evaluate_models
from src.utils.yaml_loader import load_model_config


class ModelTrainingService:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.catboost_training_dir, exist_ok=True)

    def train_and_validate(
        self, model_name, train_array: np.ndarray, test_array: np.ndarray
    ):
        try:
            # Train the model
            self.logger.info("Starting model training.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],  # Target for training
                test_array[:, :-1],  # Features for testing
                test_array[:, -1],  # Target for testing
            )

            model_configs = load_model_config()
            # Initialize models and their parameters
            models = {}
            params = {}

            for model_type, model_info in model_configs["models"].items():
                try:
                    model_class = model_info[
                        "type"
                    ]  # Extract class name (e.g., 'RandomForestRegressor')
                    model_params = model_info.get("params", {})  # Extract parameters

                    # Dynamically instantiate the model using `eval`
                    models[model_type] = eval(model_class)(**model_params)
                    params[model_type] = model_params

                    logging.info(
                        f"Loaded model: {model_name} with params: {model_params}"
                    )
                except Exception as e:
                    logging.error(f"Failed to initialize model {model_type}: {e}")
                    raise CustomException(e)

            # Evaluate all models
            logging.info(
                "Evaluating models with the provided training and testing data."
            )
            # Log the start of evaluation for the current model
            logging.info(f"Evaluating model: {model_name}")
            model = models.get(model_name, {})
            hyper_params = params.get(model_name, {})

            # Perform GridSearchCV if hyperparameters are provided
            if hyper_params:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=hyper_params,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=1,
                )
                gs.fit(X_train, y_train)

                # Update the model with the best parameters
                model.set_params(**gs.best_params_)

            # Train the model
            model.fit(X_train, y_train)

            # Predictions and scoring
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log scores for the model
            logging.info(
                f"Model: {model_name} | Train R2: {train_model_score:.4f} | Test R2: {test_model_score:.4f}"
            )

            train_results = {
                "model": model,
                "model_name": model_name,
                "train_r2": train_model_score,
                "test_r2": test_model_score,
            }

            self.logger.info("Model training completed successfully.")

            return train_results
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise
