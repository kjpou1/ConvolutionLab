import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.data_transformation_config import DataTransformationConfig
from src.utils.file_utils import save_object

logging = LoggerManager.get_logger(__name__)


class DataTransformationService:
    """
    Handles data transformation, including preprocessing, feature scaling,
    and combining input and target arrays.
    """

    def __init__(self):
        """
        Initializes the DataTransformationService class with the configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(
        self, use_cyclical_encoding=False
    ) -> ColumnTransformer:
        """
        Creates a preprocessing pipeline for both numerical and categorical columns.

        Args:
            df (pd.DataFrame): The dataset used to dynamically determine features.
            use_cyclical_encoding (bool): If True, applies cyclic transformations instead of One-Hot Encoding.

        Returns:
            ColumnTransformer: A transformer object that applies the preprocessing steps.
        """
        try:

            # **🔹 Dynamically Assign Numerical & Categorical Features**
            numerical_columns = [
                "Leavitt_Projection",
                "Leavitt_Convolution",
                "LC_Slope",
                "LC_Acceleration",
                "Convolution_Probability",
                "Momentum_T-1",
                "Momentum_T-2",
                "Momentum_T-5",
                "Momentum_T-10",
                "Momentum_T-21",
                "Returns_T-1",
                "Returns_T-2",
                "Returns_T-5",
                "Returns_T-10",
                "Returns_T-21",
                "AHMA",
                "ATR",
                "Volume",
                "Open",
                "High",
                "Low",
                "Close",
            ]

            # Feature Importance
            categorical_columns = ["Day_Of_Week", "Month", "Year"]

            # **🔹 Handle Hour Column Differently Based on Encoding Choice**
            if use_cyclical_encoding:

                def cyclic_transform(X):
                    return np.c_[np.sin(2 * np.pi * X / 24), np.cos(2 * np.pi * X / 24)]

                hour_pipeline = Pipeline(
                    steps=[
                        (
                            "cyclic_transform",
                            FunctionTransformer(cyclic_transform, validate=False),
                        )
                    ]
                )
            else:
                categorical_columns.append(
                    "Hour"
                )  # If not using cyclic encoding, treat Hour as categorical

            # **🔹 Handle Missing Values & Scaling for Numerical Features**
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Fill missing values
                    ("scaler", StandardScaler()),  # Scale numerical data
                ]
            )

            # **🔹 One-Hot Encoding for Categorical Features**
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Fill missing categorical values
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),  # Encode categorical
                    (
                        "scaler",
                        StandardScaler(with_mean=False),
                    ),  # Scale encoded features
                ]
            )

            logging.info("Categorical columns: %s", categorical_columns)
            logging.info("Numerical columns: %s", numerical_columns)

            # **🔹 Apply Transformations**
            transformers = [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ]

            if use_cyclical_encoding:
                transformers.append(("hour_pipeline", hour_pipeline, ["Hour"]))

            preprocessor = ColumnTransformer(transformers)

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(
        self, train_path: str, validation_path: str, target_column: str
    ):
        """
        Reads train and test datasets, applies preprocessing, and saves the preprocessor object.

        Args:
            train_path (str): Path to the training dataset.
            validation_path (str): Path to the validation dataset.

        Returns:
            tuple: Transformed training array, validation array, and the path to the saved preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            validation_df = pd.read_csv(validation_path)

            logging.info("Read train and validation data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_validation_df = validation_df.drop(
                columns=[target_column], axis=1
            )
            target_feature_validation_df = validation_df[target_column]

            logging.info(
                "Applying preprocessing object on training and validation data."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_validation_arr = preprocessing_obj.transform(
                input_feature_validation_df
            )

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            validation_arr = np.c_[
                input_feature_validation_arr, np.array(target_feature_validation_df)
            ]

            logging.info("Preprocessing completed and object saved.")

            logging.info(
                "Shape of transformed train array: %s", input_feature_train_arr.shape
            )
            logging.info(
                "Shape of transformed test array: %s",
                input_feature_validation_arr.shape,
            )
            return (
                train_arr,
                validation_arr,
                preprocessing_obj,
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation_for_test(
        self, test_path: str, target_column: str, preprocessing_obj
    ):
        """
        Reads the test dataset, applies the saved preprocessing transformations.

        Args:
            test_path (str): Path to the test dataset.
            target_column (str): Target column name.

        Returns:
            tuple: Transformed test array and preprocessor object.
        """
        try:
            logging.info("📥 Loading test dataset.")
            test_df = pd.read_csv(test_path)

            logging.info("🔄 Applying preprocessing to test data.")
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("✅ Test data transformation complete.")

            return test_arr

        except Exception as e:
            raise CustomException(e, sys) from e
