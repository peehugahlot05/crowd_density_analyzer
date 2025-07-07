import gdown
import os

model_path = 'models/partBmodel_best.pth'

if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    print("Downloading CSRNet model from Google Drive...")
    gdown.download('https://drive.google.com/uc?id=1toFG5ZxJfzPox5ITR_ga2LxUL_fQZyb9', model_path, quiet=False)

import os
import uuid
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request
from torchvision import transforms
from ultralytics import YOLO

from model import CSRNet  # Ensure this file exists

# -------------------- Setup --------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load YOLOv8 --------------------
yolo_model = YOLO("models/yolov8s.pt")

# -------------------- Load CSRNet --------------------
csrnet_model = CSRNet().to(device)

# ✅ Load only the weights from the checkpoint
checkpoint = torch.load("models/partBmodel_best.pth", map_location=device)
csrnet_model.load_state_dict(checkpoint["state_dict"])
csrnet_model.eval()

# -------------------- CSRNet Preprocessing --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_density(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        density_map = csrnet_model(img_tensor)
    density_map = density_map.squeeze().cpu().numpy()
    count = round(density_map.sum(), 2)
    return density_map, count

def generate_heatmap(density_map, original_shape):
    heatmap_resized = cv2.resize(density_map, (original_shape[1], original_shape[0]))
    heatmap_normalized = (heatmap_resized / heatmap_resized.max() * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    return colored_heatmap

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # Read image
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ---- YOLO Inference ----
    results = yolo_model(img_bgr)[0]
    persons = [b for b, c in zip(results.boxes.xyxy, results.boxes.cls) if int(c) == 0]
    yolo_count = len(persons)
    yolo_img = results.plot()
    yolo_filename = filename.replace(".", "_yolo.")
    yolo_path = os.path.join(app.config['UPLOAD_FOLDER'], yolo_filename)
    cv2.imwrite(yolo_path, yolo_img)

    # ---- CSRNet Inference ----
    density_map, csrnet_count = predict_density(pil_img)
    heatmap = generate_heatmap(density_map, img_bgr.shape)
    heat_overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    heat_filename = filename.replace(".", "_heat.")
    heat_path = os.path.join(app.config['UPLOAD_FOLDER'], heat_filename)
    cv2.imwrite(heat_path, heat_overlay)

    # ---- Alert Logic (Only Based on CSRNet Count) ----
    if csrnet_count < 63:
        level = "Low"
        alert = "No"
    elif csrnet_count < 143:
        level = "Medium"
        alert = "No"
    else:
        level = "High"
        alert = "Yes, anomaly detected"

    return render_template("result.html",
                           original_image=filename,
                           yolo_image=yolo_filename,
                           heatmap_image=heat_filename,
                           yolo_count=yolo_count,
                           csrnet_count=csrnet_count,
                           level=level,
                           alert=alert)

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)





