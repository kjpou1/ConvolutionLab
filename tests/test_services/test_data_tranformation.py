import os
import pytest
import pandas as pd
import numpy as np
from src.services.data_transformation_service import DataTransformationService
from src.exception import CustomException
from src.utils.file_utils import load_object


@pytest.fixture
def sample_data(tmp_path):
    """Fixture to create sample train and test data."""
    train_data = {
        "math_score": [70, 80, 90],
        "writing_score": [65, 75, 85],
        "reading_score": [60, 70, 80],
        "gender": ["male", "female", "femail"],
        "race_ethnicity": ["group A", "group B", "group C"],
        "parental_level_of_education": ["bachelor's", "high school", "master's"],
        "lunch": ["standard", "free/reduced", "standard"],
        "test_preparation_course": ["none", "completed", "none"],
    }
    test_data = {
        "math_score": [50, 60],
        "writing_score": [55, 65],
        "reading_score": [50, 60],
        "gender": ["female", "male"],
        "race_ethnicity": ["group B", "group A"],
        "parental_level_of_education": ["bachelor's", "high school"],
        "lunch": ["free/reduced", "standard"],
        "test_preparation_course": ["completed", "none"],
    }

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    pd.DataFrame(train_data).to_csv(train_path, index=False)
    pd.DataFrame(test_data).to_csv(test_path, index=False)

    return train_path, test_path


def test_data_transformation(sample_data, tmp_path):
    """Test the data transformation process."""
    train_path, test_path = sample_data

    # Initialize the service
    transformation_service = DataTransformationService()

    # Set the output path for the preprocessor
    transformation_service.data_transformation_config.preprocessor_obj_file_path = str(
        tmp_path / "preprocessor.pkl"
    )

    # Perform data transformation
    train_arr, test_arr, preprocessor_path = (
        transformation_service.initiate_data_transformation(train_path, test_path)
    )

    # Verify the shape of the transformed train array
    expected_num_features = train_arr.shape[
        1
    ]  # Dynamically get the expected number of features
    assert train_arr.shape == (
        3,
        expected_num_features,
    )  # 3 rows, dynamic number of features
    assert test_arr.shape == (
        2,
        expected_num_features,
    )  # 2 rows, dynamic number of features

    # Verify the preprocessor was saved
    assert os.path.exists(preprocessor_path)

    # Verify the saved preprocessor can be loaded
    preprocessor = load_object(preprocessor_path)
    assert preprocessor is not None


def test_data_transformation_missing_columns(tmp_path):
    """Test for missing required columns."""
    train_data = {
        "math_score": [70, 80],
        "gender": ["male", "female"],
    }
    test_data = {
        "math_score": [60, 70],
        "gender": ["female", "male"],
    }

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    pd.DataFrame(train_data).to_csv(train_path, index=False)
    pd.DataFrame(test_data).to_csv(test_path, index=False)

    # Initialize the service
    transformation_service = DataTransformationService()

    with pytest.raises(CustomException):
        transformation_service.initiate_data_transformation(train_path, test_path)
