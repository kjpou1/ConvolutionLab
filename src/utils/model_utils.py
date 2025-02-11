# utils/model_utils.py

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_iterations(model, best_params):
    """
    Extracts the number of iterations or n_estimators based on the model type.
    Returns None if the information is unavailable.
    """
    iterations = None
    if isinstance(model, CatBoostClassifier):
        iterations = (
            model.get_best_iteration() if hasattr(model, "get_best_iteration") else None
        )
    elif isinstance(model, GradientBoostingClassifier):
        iterations = model.n_estimators
    elif isinstance(model, LGBMClassifier):
        iterations = (
            model.best_iteration
            if hasattr(model, "best_iteration")
            else model.n_estimators
        )
    elif isinstance(model, XGBClassifier):
        iterations = (
            model.best_iteration
            if hasattr(model, "best_iteration")
            else model.n_estimators
        )
    elif isinstance(model, RandomForestClassifier):
        iterations = model.n_estimators
    elif isinstance(model, AdaBoostClassifier):
        iterations = model.n_estimators
    elif isinstance(model, LogisticRegression):
        iterations = model.get_params().get("max_iter", None)

    # If iterations are still None, we will try to find it in best_params
    if iterations is None:
        # Check if 'iterations' or 'n_estimators' or 'max_iter' exists in best_params
        if "iterations" in best_params:
            iterations = best_params["iterations"]
        elif "n_estimators" in best_params:
            iterations = best_params["n_estimators"]
        elif "max_iter" in best_params:
            iterations = best_params["max_iter"]

    return iterations
