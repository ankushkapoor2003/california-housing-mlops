import unittest
import joblib
import os
import numpy as np

class TestModelPrediction(unittest.TestCase):
    def test_prediction(self):
        model_path = os.path.join('models', 'model.joblib')
        try:
            model = joblib.load(model_path)
            # Sample input matching training features
            sample_input = np.array([[-119, 37.83, 28.0, 583.0, 418.0, 721.0, 583.6, 12.68]])
            prediction = model.predict(sample_input)
            self.assertIsInstance(prediction, np.ndarray, "Prediction should be a NumPy array")
            self.assertEqual(len(prediction), 1, "Should return one prediction")
        except Exception as e:
            self.fail(f"Prediction failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
