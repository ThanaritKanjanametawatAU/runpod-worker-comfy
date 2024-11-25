import requests
import os
import io
from PIL import Image
import base64
import random
import json
endpoint = os.environ.get("RUNPOD_ENDPOINT")
bearer_token = os.environ.get("RUNPOD_BEARER_TOKEN")

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

imageString = convert_image_to_base64("../../models/current.jpg")
print(imageString[:100])

# Load Json
with open("ProductionWorkflow/Chrismas/ChrismasV1-api.json", "r") as file:
    body = json.load(file)

# body["25"]["inputs"]["noise_seed"] = random.randint(0, 2**10 - 1)
endpoint_body = {
    "input": {
        "workflow": body,
        "images": [
        {
            "name": "current.jpg",
            "image": imageString
        }
        ]
    }
}

response = requests.post(endpoint, json=endpoint_body, headers={"Authorization": f"Bearer {bearer_token}"})
try:
    imageString = response.json()["output"]["message"]
except:
    print(response.json())

else:
    # Decode the image string to an image Base64
    image_data = base64.b64decode(imageString)

    # Save the image to a file
    image = Image.open(io.BytesIO(image_data))
    image.show()