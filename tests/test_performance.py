import unittest
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import logging

class TestModelPerformance(unittest.TestCase):
    def setUp(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Paths to models
        self.current_model_path = os.path.join('models', 'model.joblib')
        self.previous_model_path = os.path.join('models', 'model_previous.joblib')

        # Check if current model exists
        if not os.path.exists(self.current_model_path):
            self.fail(f"Current model not found at {self.current_model_path}")

        # Load current model
        try:
            self.current_model = joblib.load(self.current_model_path)
            self.logger.info("Loaded current model successfully.")
        except Exception as e:
            self.fail(f"Failed to load current model: {e}")

        # Load previous model if it exists
        if os.path.exists(self.previous_model_path):
            try:
                self.previous_model = joblib.load(self.previous_model_path)
                self.logger.info("Loaded previous model successfully.")
            except Exception as e:
                self.fail(f"Failed to load previous model: {e}")
        else:
            self.previous_model = None
            self.logger.warning("Previous model not found. Performance comparison will be skipped.")

        # Fetch California Housing dataset
        self.housing = fetch_california_housing()
        X = self.housing.data
        y = self.housing.target

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.logger.info("Dataset split into training and testing sets.")

    def test_performance_improvement(self):
        # Make predictions with current model
        try:
            y_pred_current = self.current_model.predict(self.X_test)
            mse_current = mean_squared_error(self.y_test, y_pred_current)
            self.logger.info(f"Current Model MSE: {mse_current}")
        except Exception as e:
            self.fail(f"Current model prediction failed: {e}")

        if self.previous_model:
            # Make predictions with previous model
            try:
                y_pred_previous = self.previous_model.predict(self.X_test)
                mse_previous = mean_squared_error(self.y_test, y_pred_previous)
                self.logger.info(f"Previous Model MSE: {mse_previous}")
            except Exception as e:
                self.fail(f"Previous model prediction failed: {e}")

            # Assert that the new model has lower MSE
            self.assertLess(
                mse_current, mse_previous,
                f"New model MSE ({mse_current}) is not less than previous model MSE ({mse_previous})."
            )
        else:
            self.skipTest("Previous model not available. Skipping performance comparison.")

if __name__ == '__main__':
    unittest.main()
