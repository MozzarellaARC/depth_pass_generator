import torch
import cv2
import numpy as np

# Load the smallest MiDaS model via PyTorch Hub
model_type = "MiDaS_small"  # MiDaS v2.1 Small (fastest, smallest)
midas = torch.hub.load("intel-isl/MiDaS", model_type)  # automatically downloads the ~60 MB float32 model
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Use small-specific transforms
transform = transforms.small_transform

# Read input image
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess & predict
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)

# Upsample to original resolution
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False
).squeeze()

# Convert to depth map & save
depth = prediction.cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
cv2.imwrite("depth_output.png", depth.astype(np.uint8))
