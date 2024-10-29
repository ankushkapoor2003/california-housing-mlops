# california-housing-mlops

# California Housing MLOps CI/CD Pipeline

## Overview

This repository implements a CI/CD pipeline for deploying a machine learning model to predict California housing prices. The pipeline is designed to automatically test, validate, and deploy the model when a new version shows performance improvements over the previous version.

## Project Structure
- **`.github/workflows`**: Stores the ci-cd pipeline yaml file for testing and deploying the models.
  - `ci-cd-pipeline.yaml`: The pipeline yaml file for testing the new candidate model and automatically deploying the model if it passes the tests. 
- **`models/`**: Stores the currently deployed (model_previous) and the new candidate (model) joblib files.
  - `model.joblib`: Proposed new model for deployment.
  - `model_previous.joblib`: Currently deployed model.
- **`tests/`**: Contains test scripts for CI/CD validation before deployment.
  - `test_model_loading.py`: Loads and validates the proposed model (model.joblib).
  - `test_prediction.py`: Validates the new model’s prediction capability for a test case.
  - `test_performance.py`: Compares the performance of the new model against the previous model on same dataset for all model deployments. Stops execution if the the performance of new model is not better than the currently deployed model
- **`make_prediction.py`**: A script to make predictions with the deployed model.
- **`requirements.txt`**: Dependencies required to run the project.

## CI/CD Pipeline

### Continuous Integration (CI)

The CI process uses automated testing triggered by any change to the model code. The tests ensure that:
1. The model loads correctly.
2. The new model performs better or as well as the previous version.
3. Predictions are accurate and valid.

**Tests Summary**:
- **`test_model_loading.py`**: Loads the `model.joblib` file and checks if the model is loaded without errors.
- **`test_performance.py`**:
  - Loads `model.joblib` and `model_previous.joblib`.
  - Tests both models on the California Housing dataset's test split.
  - Calculates MSE for each model and asserts that the new model’s MSE is lower than the previous one.
- **`test_prediction.py`**:
  - Loads `model.joblib`.
  - Tests predictions on a sample input and verifies that the output matches expected data formats and dimensions.

## CI/CD Pipeline

The CI/CD pipeline has been implemented using GitHub Actions and Google Cloud Platform (GCP), with automated processes for testing, model validation, and deployment.

### Workflow Summary

1. **Trigger Conditions**:
   - The pipeline runs automatically on any `push` or `pull request` event on the `main` branch.
   - Changes to `models/model_previous.joblib` are ignored to prevent unnecessary pipeline runs when updating the previous model artifact.

2. **Job Structure**:
   - **Build and Test**: 
     - Checks out the repository, sets up Python, and installs dependencies.
     - Runs unit tests in the `tests/` directory to ensure the new model is functional and performs well.
   - **Deploy**:
     - Authenticates to GCP using a service account key stored as a GitHub secret.
     - Uploads the new model to Google Cloud Vertex AI.
     - Undeploys the existing model and deploys the new model to the specified endpoint.
     - Updates `model_previous.joblib` after successful deployment to keep a versioned backup.

### Continuous Integration (CI)

The CI step involves automated testing to ensure:
1. The model loads correctly.
2. The new model performs better or as well as the previous version.
3. Predictions are accurate and valid.

**Tests Summary**:
- **`test_model_loading.py`**: Loads the `model.joblib` file and checks if the model loads without errors.
- **`test_performance.py`**:
  - Loads `model.joblib` and `model_previous.joblib`.
  - Tests both models on the California Housing dataset's test split.
  - Calculates MSE for each model and asserts that the new model’s MSE is lower than the previous one.
- **`test_prediction.py`**:
  - Loads `model.joblib`.
  - Tests predictions on a sample input and verifies that the output matches expected data formats and dimensions.

### Continuous Deployment (CD)

If the tests pass and the new model outperforms the previous model, the CD pipeline deploys it to production:

1. **Model Upload**: The pipeline uploads the new model artifact to Google Cloud Storage.
2. **Endpoint Update**:
   - The pipeline retrieves the newly uploaded model ID.
   - It undeploys the currently deployed model(s) from the Vertex AI endpoint and deploys the new model.
   - The deployment uses `n1-standard-4` machine types for optimal performance.
3. **Update `model_previous.joblib`**:
   - After a successful deployment, the pipeline updates `model_previous.joblib` with the latest model.
   - The updated `model_previous.joblib` is committed and pushed to the repository.

### Tools Used

- **Version Control**: Git
- **CI/CD Tool**: GitHub Actions
- **Cloud Platform**: Google Cloud Platform (GCP) for model deployment and endpoint management
- **Model Serialization**: `joblib` for saving/loading model artifacts
- **Google Cloud SDK**: Used to interact with Vertex AI for model deployment

### Prediction Workflow

The `make_prediction.py` script leverages Google Cloud’s AI Platform to make predictions using the deployed model endpoint. It demonstrates a sample prediction using predefined input features, with error handling for failed predictions.

```python
from google.cloud import aiplatform
import json

# Initialize the AI Platform client
aiplatform.init(project='california-dataset-001', location='us-central1')

# Replace with your actual Endpoint ID
endpoint = aiplatform.Endpoint('projects/california-dataset-001/locations/us-central1/endpoints/6780961986889908224')

# Define input data
instances = [
    [-119, 37.83, 28.0, 583.0, 418.0, 721.0, 583.6, 12.68],
]

# Make prediction
response = endpoint.predict(instances=instances)
print("Predictions:", response.predictions)
