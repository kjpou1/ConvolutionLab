import json
import os
from datetime import datetime

from src.config.config import Config
from src.exception import CustomException


def append_training_history(history: dict):
    """
    Appends a new training history entry to the Config-defined training history file.

    Args:
        history (dict): A dictionary containing the training history.
    """
    config = Config()
    history_file = config.HISTORY_FILE_PATH

    try:
        # Load existing history
        with open(history_file, "r", encoding="utf-8") as f:
            training_history = json.load(f)

        # Append the new history
        training_history.append(history)

        # Save the updated history
        with open(history_file, "w") as f:
            json.dump(training_history, f, indent=4)

    except Exception as e:
        raise CustomException(f"Failed to append training history.") from e


def load_training_history():
    """
    Loads the entire training history from the Config-defined training history file.

    Returns:
        list: A list of all training history entries.
    """
    config = Config()
    history_file = config.HISTORY_FILE_PATH

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(f"Failed to load training history.") from e


def update_training_history(history_entry):
    """
    Updates the centralized training history file with a new history entry.

    Args:
        history_entry (dict): The new history entry to add.
    """
    try:
        # Get the history file path from Config
        history_file_path = Config().HISTORY_FILE_PATH

        # Check if the file exists
        if os.path.exists(history_file_path):
            # Load existing history
            with open(history_file_path, "r", encoding="utf-8") as file:
                history = json.load(file)
        else:
            # Initialize an empty history if the file doesn't exist
            history = []

        # Append the new history entry
        history.append(history_entry)

        # Save the updated history back to the file
        with open(history_file_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

    except Exception as e:
        raise CustomException(e) from e
