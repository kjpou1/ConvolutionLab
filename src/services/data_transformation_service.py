import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.models.data_transformation_config import DataTransformationConfig
from src.utils.file_utils import save_object
from src.logger_manager import LoggerManager

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

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates a preprocessing pipeline for both numerical and categorical columns.

        Returns:
            ColumnTransformer: A transformer object that applies the preprocessing steps.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Categorical columns: %s", categorical_columns)
            logging.info("Numerical columns: %s", numerical_columns)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train and test datasets, applies preprocessing, and saves the preprocessor object.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            tuple: Transformed training array, testing array, and the path to the saved preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data.")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed and object saved.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info(
                "Shape of transformed train array: %s", input_feature_train_arr.shape
            )
            logging.info(
                "Shape of transformed test array: %s", input_feature_test_arr.shape
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
