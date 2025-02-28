import logging

import numpy as np

from src.exception import CustomException

logging = logging.getLogger(__name__)


class ModelTestingService:
    """
    Handles model testing and predictions on test data.
    """

    def __init__(self, model):
        """
        Initializes the ModelTestingService with a trained model.

        Args:
            model: The trained machine learning model.
        """
        self.model = model

    def run_predictions(self, test_arr):
        """
        Runs predictions on the given test dataset.

        Args:
            test_arr (np.ndarray): Transformed test dataset (features + target).

        Returns:
            tuple: (y_test, y_pred) where:
                   - y_test is the actual labels from the test dataset.
                   - y_pred is the model's predicted values.
        """
        try:
            logging.info("📡 Running predictions on test data.")

            # Extract features (X) and target labels (y)
            y_test = test_arr[:, -1]  # Assuming last column is the target
            X_test = test_arr[:, :-1]  # All other columns are features

            # Run model inference
            y_pred = self.model.predict(X_test)

            logging.info("✅ Predictions complete.")
            return y_test, y_pred

        except Exception as e:
            logging.error(f"❌ Error during model testing: {e}")
            raise CustomException(e) from e
