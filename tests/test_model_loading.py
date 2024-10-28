import unittest
import joblib
import os

class TestModelLoading(unittest.TestCase):
    def test_model_loading(self):
        model_path = os.path.join('models', 'model.joblib')
        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "Model should not be None")
        except Exception as e:
            self.fail(f"Model loading failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
