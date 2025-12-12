import json
import base64

# Path to JSON file
json_path = "/Users/kuldeeppatel/Downloads/output1.json"

# Load JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# Extract values
prediction = data["prediction"]
confidence = data["confidence"]
heatmap_b64 = data["heatmap_base64"]

print("Prediction:", prediction)
print("Confidence:", confidence)

# Decode Base64 image
img_bytes = base64.b64decode(heatmap_b64)

# Save heatmap image
with open("decoded_heatmap.jpg", "wb") as img_file:
    img_file.write(img_bytes)

print("Heatmap saved as decoded_heatmap.jpg")
