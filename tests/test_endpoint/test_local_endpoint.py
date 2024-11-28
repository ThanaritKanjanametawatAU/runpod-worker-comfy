import requests
import json
import base64
import io
from PIL import Image

endpoint = "http://localhost:8000/runsync"

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_endpoint(workflow_path, image_path=None):
    # Read the workflow JSON file
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    # Prepare the base endpoint body
    endpoint_body = {
        "input": {
            "workflow": workflow
        }
    }
    
    # Add image if provided
    if image_path:     
        print(f"Using image: {image_path}")
        try:
            image_string = convert_image_to_base64(image_path)
            endpoint_body["input"]["images"] = [
                {
                    "name": "current.jpg",
                    "image": image_string
                }
            ]
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return

    try:
        # Make the API request
        response = requests.post(endpoint, json=endpoint_body)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        response_data = response.json()
        image_string = response_data["output"]["message"]
        
        # Decode and display the image
        image_data = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image_data))
        image.show()
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
    except KeyError:
        print("Error: Unexpected response format")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Response content: {response.text}")

# Define test cases
tests = {
    "DevBase": {
        "workflow_path": "ProductionWorkflow/Base/devbaseV1-api.json"
    },
    "MooDeng": {
        "workflow_path": "ProductionWorkflow/MooDeng/MooDengV1-api.json",
        "image_path": "../../models/current.jpg"
    },
    "Chrismas": {
        "workflow_path": "ProductionWorkflow/Chrismas/ChrismasV1-api.json",
        "image_path": "../../models/current.jpg"
    }
}

# Run tests
for test_name, params in tests.items():
    print(f"\nTesting {test_name}:")
    test_endpoint(**params)