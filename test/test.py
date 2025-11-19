import requests
import pytest

API_URL = "http://localhost:8000/api/v2/upscale/"
IMAGE_PATH = "ivy_plant.jpg"  # Replace with the path to your test PNG image
SCALE_FACTOR = 4

def test_api():
    with open(IMAGE_PATH, "rb") as image_file:
        files = {"file": ("ivy_plant.png", image_file, "image/png")}
        params = {"scale_factor": SCALE_FACTOR}
        
        response = requests.post(API_URL, files=files, params=params)

        assert response.status_code == 200
        assert response.content != {"detail":"Not Found"}

        return response.content

# This file is not meant to be ran unless you want to check image diff
# To test to see if the enpoint works
# `pytest test.py`
# From within this test directory
if __name__ == "__main__":
    upscaled_image_content = test_api()

    with open("upscaled_api_image.png", "wb") as out_file:
        out_file.write(upscaled_image_content)
        print("Upscaled image saved as 'upscaled_api_image.png'")
