from google.cloud import aiplatform
import json

# Initialize the AI Platform client
aiplatform.init(project='california-dataset-001', location='us-central1')

# Replace with your actual Endpoint ID
endpoint = aiplatform.Endpoint('projects/california-dataset-001/locations/us-central1/endpoints/6780961986889908224')

# Define your input data in the correct format (list of lists)
instances = [
    [-119, 37.83, 28.0, 583.0, 418.0, 721.0, 583.6, 12.68], 
]

# Make the prediction
try:
    response = endpoint.predict(instances=instances)
    print("Predictions:", response.predictions)
except Exception as e:
    print("Prediction failed:", e)
