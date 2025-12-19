import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Import your custom architecture
from model_architecture import CheXpertModel

# ==========================================
# 1. CONFIGURATION (FIXED PATHS)
# ==========================================
# Base folder based on your screenshot
BASE_DIR = 'data/CheXpert-v1.0-small'

# Paths to the split CSVs you created
TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")

# Root directory for images
IMAGE_ROOT = 'data' 

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters for RTX 3050 (4GB)
IMG_SIZE = 224
BATCH_SIZE = 16        
ACCUM_STEPS = 2        
NUM_WORKERS = 2        
EPOCHS_WARMUP = 1      
EPOCHS_FINETUNE = 4    
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {DEVICE}")
print(f"Training Data: {TRAIN_CSV}")

# The 14 target pathologies
LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# ==========================================
# 2. DATASET CLASS
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
        
        # Path Correction Logic:
        # If your CSV says "CheXpert-v1.0-small/train/..." 
        # and your root_dir is "data", 
        # then join("data", "CheXpert-v1.0-small/train/...") works perfectly.
        
        img_name = os.path.join(self.root_dir, rel_path)
        
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # Create black image if file missing (prevents crash)
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx][LABELS].values.astype('float32')
        return image, torch.tensor(labels)

def get_pos_weights(dataset):
    # Calculate class weights to handle imbalance (e.g., Fracture)
    df = dataset.data
    pos_weights = []
    for label in LABELS:
        n_pos = (df[label] == 1).sum()
        n_neg = (df[label] == 0).sum()
        # prevent div by zero
        if n_pos == 0: n_pos = 1 
        weight = n_neg / n_pos
        pos_weights.append(weight)
    return torch.tensor(pos_weights).to(DEVICE)

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch_idx, is_frozen=False):
    model.train()
    running_loss = 0.0
    
    desc = "Warmup" if is_frozen else "Fine-tune"
    loop = tqdm(loader, desc=f"Epoch {epoch_idx+1} [{desc}]")
    
    for i, (images, labels) in enumerate(loop):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / ACCUM_STEPS 

        # Scaled Backward
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * ACCUM_STEPS
        loop.set_postfix(loss=loss.item() * ACCUM_STEPS)
        
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
    return running_loss / len(loader)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Stats from your preprocessing step
    mean = [0.506, 0.506, 0.506]
    std =  [0.290, 0.290, 0.290]
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # --- Load Data ---
    print("Loading Data...")
    
    # IMPORTANT: IMAGE_ROOT is 'data' because your CSV paths likely include 
    # 'CheXpert-v1.0-small/train/...' 
    # So 'data' + 'CheXpert.../train/...' = Valid Path
    train_ds = ChexpertDataset(TRAIN_CSV, root_dir=IMAGE_ROOT, transform=transform)
    val_ds = ChexpertDataset(VAL_CSV, root_dir=IMAGE_ROOT, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # --- Initialize Model ---
    print("Initializing Model...")
    model = CheXpertModel(num_classes=14, backbone_name='densenet121', pretrained=True)
    model = model.to(DEVICE)
    
    # --- Optimization Engine ---
    pos_weights = get_pos_weights(train_ds)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler() 
    
    # ==========================
    # PHASE 1: WARMUP (Frozen)
    # ==========================
    print("\n--- PHASE 1: WARMUP (Frozen Backbone) ---")
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    for epoch in range(EPOCHS_WARMUP):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, is_frozen=True)
        val_loss = validate(model, val_loader, criterion)
        print(f"Warmup Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        
    # ==========================
    # PHASE 2: FINE-TUNING
    # ==========================
    print("\n--- PHASE 2: FINE-TUNING (Unfrozen Backbone) ---")
    
    for param in model.features.parameters():
        param.requires_grad = True
        
    # Lower LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE / 10
        
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS_FINETUNE):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, is_frozen=False)
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Fine-tune Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, "chexpert_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Best Model Saved: {save_path}")
            
    print("\nTraining Complete.")
