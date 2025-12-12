from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io
import numpy as np
import cv2
import base64

# ----------------------------
# LOAD MODEL
# ----------------------------

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),

            nn.Linear(128*28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# DEVICE SETUP
# ----------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ----------------------------
# LOAD SAVED MODEL
# ----------------------------

model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
model.eval()

# ----------------------------
# TRANSFORM
# ----------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

import numpy as np
import cv2
import base64

# ----------------------------
# GRAD-CAM IMPLEMENTATION
# ----------------------------

def generate_gradcam(model, img_tensor, target_layer):
    gradients = []
    activations = []

    # Hook for gradients
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Hook for forward activations
    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward  = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(img_tensor)
    pred = output.item()

    # Backward pass for Grad-CAM
    model.zero_grad()
    output.backward()

    # Remove hooks
    handle_backward.remove()
    handle_forward.remove()

    # Get stored values
    grad = gradients[0]              # [batch, channels, H, W]
    act = activations[0]             # [batch, channels, H, W]

    weights = torch.mean(grad, dim=(2,3), keepdim=True)   # Global average pooling
    cam = (weights * act).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam / cam.max()  # Normalize to 0–1

    cam = cam.detach().cpu().numpy()


    # Resize heatmap to 224x224
    heatmap = cv2.resize(cam, (224,224))

    return heatmap, pred

# FASTAPI APP

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("L") #L is for grayscale, RGB for color

        # Transform to tensor
        tensor_img = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(tensor_img).item()

        result = "Tumor" if pred > 0.5 else "No Tumor"

        return {
            "prediction": result,
            "confidence": float(pred)
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict_with_heatmap")
async def predict_with_heatmap(file: UploadFile = File(...)):
    try:
        # Load image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        tensor_img = transform(img).unsqueeze(0).to(device)

        # Target layer (last conv layer)
        target_layer = model.model[8]  # Conv2d(64,128,...)

        heatmap, pred = generate_gradcam(model, tensor_img, target_layer)
        
        # Create heatmap overlay on original image
        img_np = np.array(img.resize((224,224)))
        heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        overlay = 0.4 * heatmap_color + 0.6 * cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        overlay = overlay.astype(np.uint8)

        # Encode image as Base64
        _, buffer = cv2.imencode('.jpg', overlay)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        result = "Tumor" if pred > 0.5 else "No Tumor"

        return {
            "prediction": result,
            "confidence": float(pred),
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

