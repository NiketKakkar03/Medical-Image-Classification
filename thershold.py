import torch
import numpy as np
from sklearn.metrics import roc_curve, f1_score
import pandas as pd
import json
from model_architecture import CheXpertModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# ==========================================
# CONFIGURATION (Same as evaluate.py)
# ==========================================
BASE_DIR = 'data/CheXpert-v1.0-small'
VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")
MODEL_PATH = "models/chexpert_best.pth"
IMAGE_ROOT = 'data'
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

class ChexpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        rel_path = self.data.iloc[idx]['Path']
        img_name = os.path.join(self.root_dir, rel_path)
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            image = self.transform(image)
        labels = self.data.iloc[idx][LABELS].values.astype('float32')
        return image, torch.tensor(labels)

# Load model and data
print("Loading model...")
model = CheXpertModel(num_classes=14, backbone_name='densenet121', pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.506, 0.506, 0.506], [0.290, 0.290, 0.290])
])

val_ds = ChexpertDataset(VAL_CSV, root_dir=IMAGE_ROOT, transform=transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Get all predictions and labels
print("Running inference...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_labels)

# Calculate optimal threshold for each disease
print("\n" + "="*50)
print("Optimal Thresholds (F1-Score Maximization)")
print("="*50)

thresholds_dict = {}

for i, label in enumerate(LABELS):
    # Use ROC curve to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred_binary = (y_pred[:, i] >= threshold).astype(int)
        f1 = f1_score(y_true[:, i], y_pred_binary, zero_division=0)
        f1_scores.append(f1)
    
    # Find threshold with max F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    thresholds_dict[label] = float(optimal_threshold)
    print(f"{label:<30} | Threshold: {optimal_threshold:.4f} | F1: {optimal_f1:.4f}")

# Save thresholds to JSON
with open("thresholds.json", "w") as f:
    json.dump(thresholds_dict, f, indent=4)

print(f"\nThresholds saved to thresholds.json")
