import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from PIL import Image
from torch.utils.data import Dataset

# Import your architecture
from model_architecture import CheXpertModel

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = 'data/CheXpert-v1.0-small'
VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")
MODEL_PATH = "models/chexpert_best.pth"
IMAGE_ROOT = 'data' 
IMG_SIZE = 224
BATCH_SIZE = 32 # Validation fits more in memory (no gradients)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# ==========================================
# DATASET CLASS (Same as Train)
# ==========================================
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

# ==========================================
# EVALUATION LOGIC
# ==========================================
def evaluate():
    print(f"Using Device: {DEVICE}")
    
    # 1. Setup Data
    mean = [0.506, 0.506, 0.506]
    std =  [0.290, 0.290, 0.290]
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_ds = ChexpertDataset(VAL_CSV, root_dir=IMAGE_ROOT, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Load Model
    print(f"Loading Model: {MODEL_PATH} ...")
    model = CheXpertModel(num_classes=14, backbone_name='densenet121', pretrained=False)
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    # 3. Inference Loop
    print("Running Inference on Validation Set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(DEVICE)
            
            # Forward Pass
            # Note: Model returns logits (raw scores). We need Probabilities for AUC.
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    # 4. Calculate AUC for each Class
    print("\n" + "="*40)
    print(f"{'Pathology':<30} | {'AUC Score':<10}")
    print("="*40)
    
    aucs = []
    for i, label in enumerate(LABELS):
        # Handle case where a class has only 0s or only 1s in val set (rare but possible)
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            print(f"{label:<30} | {score:.4f}")
            aucs.append(score)
        except ValueError:
            print(f"{label:<30} | N/A (Only one class present)")
            
    print("-" * 40)
    print(f"{'AVERAGE AUC':<30} | {np.mean(aucs):.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()
