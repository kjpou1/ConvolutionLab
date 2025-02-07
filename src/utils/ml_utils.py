import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models with hyperparameter tuning using GridSearchCV.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        models (dict): Dictionary of model names and their instances.
        param (dict): Dictionary of model names and their hyperparameter grids.

    Returns:
        dict: A dictionary with model names as keys and test R2 scores as values.
    """
    try:
        report = {}

        for model_name, model in models.items():
            try:
                # Log the start of evaluation for the current model
                logging.info(f"Evaluating model: {model_name}")
                para = param.get(model_name, {})

                # Perform GridSearchCV if hyperparameters are provided
                if para:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=para,
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

                report[model_name] = test_model_score

                # Log scores for the model
                logging.info(
                    f"Model: {model_name} | Train R2: {train_model_score:.4f} | Test R2: {test_model_score:.4f}"
                )
            except Exception as model_error:
                # Log an error if the specific model fails during evaluation
                logging.error(f"Error evaluating model {model_name}: {model_error}")
                report[model_name] = None

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
