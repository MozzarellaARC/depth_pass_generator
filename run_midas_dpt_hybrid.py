import torch
import cv2
import numpy as np

# Load model from PyTorch Hub
model_type = "DPT_Hybrid"  # Larger than MiDaS_small, but much better quality
midas = torch.hub.load("intel-isl/MiDaS", model_type)
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Use the correct transform
transform = transforms.dpt_transform

# Load and process input image
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)

# Resize to original image size
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze() 

# Normalize and save depth
depth = prediction.cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
cv2.imwrite("depth_output.png", depth.astype(np.uint8))
