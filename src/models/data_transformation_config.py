import os
from dataclasses import dataclass

from src.config.config import Config


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    Defines the file path for saving the preprocessor object.
    """

    preprocessor_obj_file_path: str = Config().PREPROCESSOR_FILE_PATH
