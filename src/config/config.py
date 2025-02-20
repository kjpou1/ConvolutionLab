import os

from dotenv import load_dotenv

from src.models import SingletonMeta


class Config(metaclass=SingletonMeta):
    """
    Singleton configuration class for managing project-wide constants and directories.
    """

    _is_initialized = False  # Tracks whether the Config has already been initialized

    def __init__(self):
        # Prevent re-initialization
        if Config._is_initialized:
            return

        # Load environment variables
        load_dotenv()

        # Attributes that can be set dynamically
        self._debug = os.getenv("DEBUG", False)
        self._config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
        self._model_type = None  # Default value for model_type
        self._best_of_all = False  # Default value for best_of_all
        self._save_best = False  # Default value for save_best
        self._input_data_file_path = os.getenv("INPUT_DATA_FILE_PATH", "data/data.csv")
        self._target_column = os.getenv(
            "TARGET_COLUMN", "Target_T+1"
        )  # Default target column

        self._model_config_path = os.getenv(
            "MODEL_CONFIG_FILE_PATH", "config/model_config.yml"
        )

        # Load Leavitt-related parameters
        self._ahma_window = int(os.getenv("AHMA_WINDOW", 20))
        self._leavitt_proj_window = int(os.getenv("LEAVITT_PROJ_WINDOW", 9))
        self._leavitt_conv_window = int(os.getenv("LEAVITT_CONV_WINDOW", 5))

        # ATR Window
        self._atr_window = int(os.getenv("ATR_WINDOW", 14))

        # Movement Classification Parameters
        self._movement_period = int(os.getenv("MOVEMENT_PERIOD", 14))
        self._movement_scale_factor = float(os.getenv("MOVEMENT_SCALE_FACTOR", 0.25))

        # Base directory for artifacts
        self.BASE_DIR = os.getenv("BASE_DIR", "artifacts")

        # Subdirectories for artifacts
        self.RAW_DATA_DIR = os.path.join(self.BASE_DIR, "data", "raw")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.MODEL_FILE_PATH = os.path.join(self.MODEL_DIR, "model.pkl")
        self.CONFIDENCE_RANGE_FILE_PATH = os.path.join(
            self.MODEL_DIR, "confidence_range.pkl"
        )
        self.PREPROCESSOR_FILE_PATH = os.path.join(self.BASE_DIR, "preprocessor.pkl")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.HISTORY_DIR = os.path.join(self.BASE_DIR, "history")
        self.HISTORY_FILE_PATH = os.path.join(
            self.BASE_DIR, "history", "training_history.json"
        )
        self.REPORTS_DIR = os.path.join(self.BASE_DIR, "reports")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_DIR, "data", "processed")

        # Ensure all necessary directories exist
        self._ensure_directories_exist()

        # Mark as initialized
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        """
        Ensures that all necessary directories exist. Creates them if they do not.
        """
        directories = [
            self.RAW_DATA_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
            self.REPORTS_DIR,
            self.HISTORY_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @property
    def target_column(self):
        """Get the target column."""
        return self._target_column

    @target_column.setter
    def target_column(self, value):
        """Set the target column."""
        if not isinstance(value, str):
            raise ValueError("target_column must be a string.")
        self._target_column = value

    @property
    def config_path(self):
        """Get the configuration file path."""
        return self._config_path

    @config_path.setter
    def config_path(self, value):
        """Set the configuration file path."""
        if not isinstance(value, str):
            raise ValueError("config_path must be a string.")
        self._config_path = value

    @property
    def debug(self):
        """Get the debug mode status."""
        return self._debug

    @debug.setter
    def debug(self, value):
        """Set the debug mode status."""
        if not isinstance(value, bool):
            raise ValueError("debug must be a boolean value.")
        self._debug = value

    @property
    def model_type(self):
        """Get the model type(s) for training."""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        """Set the model type(s) for training."""
        if not isinstance(value, (list, type(None))):
            raise ValueError("model_type must be a list or None.")
        self._model_type = value

    @property
    def best_of_all(self):
        """Get the best_of_all flag."""
        return self._best_of_all

    @best_of_all.setter
    def best_of_all(self, value):
        """Set the best_of_all flag."""
        if not isinstance(value, bool):
            raise ValueError("best_of_all must be a boolean value.")
        self._best_of_all = value

    @property
    def save_best(self):
        """Get the save_best flag."""
        return self._save_best

    @save_best.setter
    def save_best(self, value):
        """Set the save_best flag."""
        if not isinstance(value, bool):
            raise ValueError("save_best must be a boolean value.")
        self._save_best = value

    @property
    def input_data_file_path(self):
        """Get the input data file path."""
        return self._input_data_file_path

    @input_data_file_path.setter
    def input_data_file_path(self, value):
        """Set the input data file path."""
        if not isinstance(value, str):
            raise ValueError("input_data_file_path must be a string.")
        self._input_data_file_path = value

    # Getter and Setter for AHMA_WINDOW
    @property
    def ahma_window(self):
        """Get AHMA window size."""
        return self._ahma_window

    @ahma_window.setter
    def ahma_window(self, value):
        """Set AHMA window size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("AHMA window must be a positive integer.")
        self._ahma_window = value

    # Getter and Setter for LEAVITT_PROJ_WINDOW
    @property
    def leavitt_proj_window(self):
        """Get Leavitt Projection window size."""
        return self._leavitt_proj_window

    @leavitt_proj_window.setter
    def leavitt_proj_window(self, value):
        """Set Leavitt Projection window size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Leavitt Projection window must be a positive integer.")
        self._leavitt_proj_window = value

    # Getter and Setter for LEAVITT_CONV_WINDOW
    @property
    def leavitt_conv_window(self):
        """Get Leavitt Convolution window size."""
        return self._leavitt_conv_window

    @leavitt_conv_window.setter
    def leavitt_conv_window(self, value):
        """Set Leavitt Convolution window size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Leavitt Convolution window must be a positive integer.")
        self._leavitt_conv_window = value

    @property
    def atr_window(self):
        """Get ATR window size."""
        return self._atr_window

    @atr_window.setter
    def atr_window(self, value):
        """Set ATR window size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ATR window must be a positive integer.")
        self._atr_window = value

    @property
    def movement_period(self):
        """Get movement classification period."""
        return self._movement_period

    @movement_period.setter
    def movement_period(self, value):
        """Set movement classification period."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Movement period must be a positive integer.")
        self._movement_period = value

    @property
    def movement_scale_factor(self):
        """Get scale factor for movement classification."""
        return self._movement_scale_factor

    @movement_scale_factor.setter
    def movement_scale_factor(self, value):
        """Set scale factor for movement classification."""
        if not isinstance(value, float) or value <= 0:
            raise ValueError("Movement scale factor must be a positive float.")
        self._movement_scale_factor = value

    @property
    def model_file(self):
        """Get the model filename."""
        return self._model_file

    @model_file.setter
    def model_file(self, value):
        """Set the model filename."""
        if not isinstance(value, str):
            raise ValueError("model_file must be a string.")
        self._model_file = value
        self.MODEL_FILE_PATH = os.path.join(self.MODEL_DIR, self._model_file)

    @property
    def model_config_path(self):
        """Get the model configuration file path."""
        return self._model_config_path

    @model_config_path.setter
    def model_config_path(self, value):
        """Set the model configuration file path."""
        if not isinstance(value, str):
            raise ValueError("model_config_file_path must be a string.")
        self._model_config_path = value

    @classmethod
    def initialize(cls):
        """
        Explicitly initializes the Config singleton.
        This ensures that the configuration is set up before being used in the application.
        """
        if not cls._is_initialized:
            cls()

    @classmethod
    def is_initialized(cls):
        """
        Checks whether the Config singleton has been initialized.
        Returns:
            bool: True if initialized, False otherwise.
        """
        return cls._is_initialized

    @classmethod
    def reset(cls):
        """
        Resets the Config singleton for testing purposes.
        """
        cls._is_initialized = False
        cls._instances = {}
