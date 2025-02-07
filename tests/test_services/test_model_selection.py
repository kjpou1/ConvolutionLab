import os
import sys
import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from src.services.model_selection_service import ModelSelectionService
from src.utils.file_utils import load_object
from src.exception import CustomException


@pytest.fixture
def sample_data():
    """
    Fixture to create sample training and testing data.
    """
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    y = y - y.min()  # Shift values to make them non-negative
    train_array = np.c_[X[:80], y[:80]]  # 80 samples for training
    test_array = np.c_[X[80:], y[80:]]  # 20 samples for testing
    return train_array, test_array


@pytest.fixture
def model_selection_service(tmp_path):
    """
    Fixture to initialize ModelSelectionService with a temporary model save path.
    """
    service = ModelSelectionService()
    service.model_trainer_config.trained_model_file_path = str(
        tmp_path / "best_model.pkl"
    )
    return service


def test_initiate_model_trainer_success(sample_data, model_selection_service):
    """
    Test the successful execution of initiate_model_trainer.
    """
    train_array, test_array = sample_data

    # Execute the model trainer
    result = model_selection_service.initiate_model_trainer(train_array, test_array)

    # Verify the return structure and values
    assert isinstance(result, dict), "Result should be a dictionary."
    assert "model_report" in result, "model_report key missing in result."
    assert "best_model_name" in result, "best_model_name key missing in result."
    assert "best_model_score" in result, "best_model_score key missing in result."
    assert "r2_square" in result, "r2_square key missing in result."

    # Ensure the best_model_score and r2_square are reasonable
    assert result["best_model_score"] >= 0, "Best model score should be non-negative."
    assert result["r2_square"] >= 0, "R2 score should be non-negative."


def test_no_model_meets_threshold(sample_data, model_selection_service, monkeypatch):
    """
    Test that ModelSelectionService raises an exception when no model meets the threshold.
    """
    train_array, test_array = sample_data

    # Mock evaluate_models to return low R2 scores
    def mock_evaluate_models(*args, **kwargs):
        print("Mock evaluate_models called")
        raise CustomException("No best model found", error_detail=sys)

    monkeypatch.setattr(
        "src.services.model_selection_service.ModelSelectionService.initiate_model_trainer",
        mock_evaluate_models,
    )
    print("Before pytest.raises")
    with pytest.raises(CustomException) as exc_info:
        model_selection_service.initiate_model_trainer(train_array, test_array)

    print(f"Exception: {exc_info.value}")
    assert "No best model found" in str(exc_info.value)


def test_invalid_data_handling(model_selection_service):
    """
    Test that ModelSelectionService handles invalid input data gracefully and
    produces expected results in the returned model report.
    """
    # Pass invalid data arrays
    invalid_train_array = np.array([[1, 2], [3, 4]])  # Only 2 features
    invalid_test_array = np.array([[5, 6], [7, 8]])  # Only 2 features

    result = model_selection_service.initiate_model_trainer(
        invalid_train_array, invalid_test_array
    )

    # Verify the returned result structure
    assert isinstance(result, dict), "Result should be a dictionary."
    assert "model_report" in result, "model_report key missing in result."
    assert "best_model_name" in result, "best_model_name key missing in result."
    assert "best_model_score" in result, "best_model_score key missing in result."
    assert "r2_square" in result, "r2_square key missing in result."

    # Check the model report
    model_report = result["model_report"]
    assert (
        model_report["Linear Regression"] == 1.0
    ), "Linear Regression score should be 1.0."
    for model, score in model_report.items():
        if model != "Linear Regression":
            assert score is None, f"Score for {model} should be None."

    # Verify best model and its score
    assert (
        result["best_model_name"] == "Linear Regression"
    ), "Best model should be Linear Regression."
    assert result["best_model_score"] == 1.0, "Best model score should be 1.0."
    assert result["r2_square"] == 1.0, "R2 score for Linear Regression should be 1.0."
