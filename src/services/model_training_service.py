import logging
import os

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config.config import Config
from src.exception import CustomException
from src.models.model_trainer_config import ModelTrainerConfig
from src.utils.yaml_loader import load_model_config


class ModelTrainingService:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.catboost_training_dir, exist_ok=True)

    def train_and_validate(
        self, model_name, train_array: np.ndarray, validation_array: np.ndarray
    ):
        try:
            # Train the model
            self.logger.info("Starting model training.")

            X_train, y_train, X_val, y_val = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],  # Target (Classification Labels)
                validation_array[:, :-1],  # Features for testing
                validation_array[:, -1],  # Target for validation
            )

            # Load model configuration
            model_configs = load_model_config()

            # Extract the requested model's configuration
            if model_name not in model_configs["models"]:
                raise ValueError(f"Model {model_name} not found in configuration.")

            model_info = model_configs["models"][model_name]
            model_class = model_info["type"]
            constructor_params = model_info.get("constructor_params", {})
            model_params = model_info.get("params", {})

            # Merge constructor_params and model_params
            all_params = {**constructor_params, **model_params}

            # Initialize only the selected model
            try:
                model = eval(model_class)(**all_params)
                logging.info(f"Loaded model: {model_name} with params: {model_params}")
            except Exception as e:
                logging.error(f"Failed to initialize model {model_name}: {e}")
                raise CustomException(e)

            # Extract hyperparameters for GridSearchCV (if available)
            hyper_params = model_info.get("params", {})

            best_params = None
            best_val_accuracy = None

            # Perform GridSearchCV if hyperparameters are provided
            if hyper_params:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=hyper_params,
                    cv=3,
                    scoring="accuracy",  # Classification metric
                    n_jobs=-1,
                    verbose=1,
                )
                gs.fit(X_train, y_train)

                # Combine constructor_params and best_params from GridSearchCV
                if gs.best_params_:
                    all_params = {**constructor_params, **gs.best_params_}

                    # Re-instantiate the model with best parameters and retrain
                    model = eval(model_class)(**all_params)
                    model.fit(X_train, y_train)

                # Store the best parameters and validation accuracy
                best_params = gs.best_params_
                best_val_accuracy = gs.best_score_

            # Predictions and scoring
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            # Calculate prediction probabilities
            if hasattr(model, "predict_proba"):
                y_val_proba = model.predict_proba(X_val)
                avg_confidence = np.mean(
                    np.max(y_val_proba, axis=1)
                )  # Take highest probability per sample
            else:
                avg_confidence = None  # If model doesn't support probability

            # Log scores for the model
            logging.info(
                f"Model: {model_name} | Train Accuracy: {train_accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f} | Avg Confidence: {avg_confidence:.4f}"
            )

            train_results = {
                "model": model,
                "model_name": model_name,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "avg_confidence": float(avg_confidence),
                "best_params": best_params,
                "best_val_accuracy": best_val_accuracy,
            }

            self.logger.info("Model training completed successfully.")
            return train_results

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise CustomException(e)
