from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from torchvision import transforms
from PIL import Image
import io
import json


from model_architecture import CheXpertModel

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "models/chexpert_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# The exact same stats from training
STATS_MEAN = [0.506, 0.506, 0.506]
STATS_STD =  [0.290, 0.290, 0.290]

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

with open("thresholds.json", "r") as f:
    THRESHOLDS = json.load(f)

# ==========================================
# APP INITIALIZATION
# ==========================================
app = FastAPI(title="CheXpert AI API", description="Medical X-Ray Classification")

# Load Model Once at Startup
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
model = CheXpertModel(num_classes=14, backbone_name='densenet121', pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval() # Set to evaluation mode (turns off Dropout)
print("Model loaded successfully.")

# Preprocessing Pipeline (Must match training!)
transform_pipeline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(STATS_MEAN, STATS_STD)
])

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"message": "CheXpert AI is running. Use /predict to classify X-rays."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Returns both raw probabilities and binary predictions (Present/Absent).
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images supported.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        input_tensor = transform_pipeline(image).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)
            
        probs_list = probabilities.cpu().numpy()[0]
        
        # Create two response formats
        predictions = {}
        predictions_binary = {}
        
        for i, label in enumerate(LABELS):
            score = float(probs_list[i])
            threshold = THRESHOLDS[label]
            
            # Raw probability
            predictions[label] = f"{score:.4f}"
            
            # Binary decision (Present/Absent)
            is_present = score >= threshold
            predictions_binary[label] = "Present" if is_present else "Absent"
            
        return {
            "filename": file.filename,
            "raw_probabilities": predictions,
            "binary_predictions": predictions_binary,
            "note": "Binary predictions use optimized thresholds. Use raw_probabilities for custom thresholds."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
