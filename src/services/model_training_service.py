import logging
import os

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config.config import Config
from src.exception import CustomException
from src.models.model_trainer_config import ModelTrainerConfig
from src.services.report_service import ReportService
from src.utils.model_utils import get_iterations
from src.utils.utils import safe_float
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
            model_info = self._get_model_info(model_name, model_configs)

            # Initialize model
            model = self._initialize_model(model_info)

            # Perform hyperparameter tuning
            model, best_params, best_val_accuracy = self.hyperparameter_tuning(
                model, model_info, X_train, y_train, X_val=X_val, y_hat=y_val
            )

            # Predictions and scoring
            metrics = self.calculate_and_log_metrics(
                model, X_train, y_train, X_val, y_val
            )

            # Get iterations (or n_estimators) for model
            iterations = get_iterations(model=model, best_params=best_params)

            # Store metrics
            performance_metrics = self.generate_performance_metrics(metrics, iterations)

            # Log the model's performance
            self.logger.info(
                f"Model: {model_name} | Train Accuracy: {metrics['train_accuracy']:.4f} | Validation Accuracy: {metrics['validation_accuracy']:.4f}"
            )

            # plot_learning_curve(
            #     model,
            #     X_train,
            #     y_train,
            #     X_val,
            #     y_val,
            #     iterations,
            #     max_iter=300,
            #     step_size=10,
            # )
            # Generate the report
            report_service = ReportService(model, X_train, y_train, X_val, y_val)
            report = report_service.generate_report()

            train_results = {
                "model": model,
                "model_name": model_name,
                "performance_metrics": performance_metrics,
                "best_params": best_params,
                "best_val_accuracy": best_val_accuracy,
                "validation_report": report,
            }

            self.logger.info("Model training completed successfully.")
            return train_results

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise CustomException(e)

    # def plot_learning_curve(
    #     model,
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     # param_name="iterations",
    #     pIterations
    #     max_iter=300,
    #     step_size=10,
    # ):
    #     """
    #     Plot learning curve by varying the number of iterations (or n_estimators).
    #     """
    #     # Set the range of iterations
    #     iterations_range = list(range(1, pIterations)) # max_iter + 1, step_size))

    #     # Store metrics for each iteration
    #     train_accuracies = []
    #     val_accuracies = []
    #     train_errors = []
    #     val_errors = []

    #     for iterations in iterations_range:
    #         # Update the model with the current iteration
    #         model.set_params(**{param_name: iterations})

    #         # Train the model with the current iteration
    #         model.fit(X_train, y_train)

    #         # Get predictions
    #         y_train_pred = model.predict(X_train)
    #         y_val_pred = model.predict(X_val)

    #         # Calculate accuracy and error
    #         train_accuracy = accuracy_score(y_train, y_train_pred)
    #         val_accuracy = accuracy_score(y_val, y_val_pred)
    #         train_error = 1 - train_accuracy
    #         val_error = 1 - val_accuracy

    #         # Store metrics
    #         train_accuracies.append(train_accuracy)
    #         val_accuracies.append(val_accuracy)
    #         train_errors.append(train_error)
    #         val_errors.append(val_error)

    #     # Plot learning curve
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(
    #         iterations_range, train_accuracies, label="Train Accuracy", color="blue"
    #     )
    #     plt.plot(
    #         iterations_range, val_accuracies, label="Validation Accuracy", color="green"
    #     )
    #     plt.xlabel(f"{param_name.capitalize()}")
    #     plt.ylabel("Accuracy")
    #     plt.title(f"Learning Curve: {param_name.capitalize()} vs Accuracy")
    #     plt.legend()
    #     plt.grid(True)
    #     # plt.show()

    #     plt.savefig("plot.png")  # Save as PNG file
    #     return train_accuracies, val_accuracies, train_errors, val_errors

    def _get_model_info(self, model_name, model_configs):
        """Helper function to extract model configuration"""
        if model_name not in model_configs["models"]:
            raise ValueError(f"Model {model_name} not found in configuration.")
        return model_configs["models"][model_name]

    def _initialize_model(self, model_info):
        """Helper function to initialize model with parameters"""
        model_class = model_info["type"]
        constructor_params = model_info.get("constructor_params", {})
        model_params = model_info.get("params", {})

        # Merge constructor_params and model_params
        all_params = {**constructor_params, **model_params}

        try:
            model = eval(model_class)(**all_params)
            self.logger.info(f"Loaded model with params: {model_params}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise CustomException(e)

    def hyperparameter_tuning(self, model, model_info, X_train, y_train, X_val, y_hat):
        """Perform hyperparameter tuning using GridSearchCV."""
        hyper_params = model_info.get("params", {})
        best_params = None
        best_val_accuracy = None

        if hyper_params:
            gs = GridSearchCV(
                estimator=model,
                param_grid=hyper_params,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
            gs.fit(
                X_train,
                y_train,
            )

            # Combine constructor_params and best_params from GridSearchCV
            constructor_params = model_info.get("constructor_params", {})
            model_class = model_info["type"]

            if gs.best_params_:
                # Re-instantiate the model with the best parameters and retrain
                all_params = {**constructor_params, **gs.best_params_}
                model = eval(model_class)(**all_params)  # Re-instantiate model
                model.fit(X_train, y_train)  # Fit the new model

            best_params = gs.best_params_
            best_val_accuracy = gs.best_score_

        return model, best_params, best_val_accuracy  # Return the updated model

    def calculate_and_log_metrics(self, model, X_train, y_train, X_val, y_val):
        """Calculate various metrics for training and validation."""
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate accuracy and error
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_error = 1 - train_accuracy
        val_error = 1 - val_accuracy

        # For multi-class classification, use average='macro' or 'weighted'
        val_f1_macro = precision_recall_fscore_support(
            y_val, y_val_pred, average="macro", zero_division=0
        )[2]
        val_f1_weighted = precision_recall_fscore_support(
            y_val, y_val_pred, average="weighted", zero_division=0
        )[2]

        # Calculate prediction probabilities if available
        avg_confidence = None
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)
            avg_confidence = np.mean(
                np.max(y_val_proba, axis=1)
            )  # Highest probability per sample

        return {
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "train_error": train_error,
            "val_error": val_error,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
            "avg_confidence": avg_confidence,
        }

    def generate_performance_metrics(self, metrics, iterations):
        """Generate and return performance metrics dictionary."""
        return {
            "train_accuracy": safe_float(metrics["train_accuracy"]),
            "validation_accuracy": safe_float(metrics["validation_accuracy"]),
            "train_error": safe_float(metrics["train_error"]),
            "val_error": safe_float(metrics["val_error"]),
            "val_f1_macro": safe_float(metrics["val_f1_macro"]),
            "val_f1_weighted": safe_float(metrics["val_f1_weighted"]),
            "avg_confidence": safe_float(metrics["avg_confidence"]),
            "iterations": safe_float(iterations),
        }
