import requests

# Test text endpoint
text_payload = {"question": "Draw the structure of benzene."}
response = requests.post("http://localhost:8000/detect_text", json=text_payload)
print("Text Detection:", response.json())

# Test image endpoint
with open("data/processed_images/sample_image.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/detect_image", files={"file": f})
print("Image Detection:", response.json())