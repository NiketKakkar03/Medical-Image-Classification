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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from sklearn.metrics import roc_auc_score

# Import your custom architecture (Keep the new MLP Head version!)
from model_architecture import CheXpertModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = 'data/CheXpert-v1.0-small'
TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_split.csv")
IMAGE_ROOT = 'data' 
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16        
ACCUM_STEPS = 2        
NUM_WORKERS = 2        
EPOCHS_WARMUP = 2      
EPOCHS_FINETUNE = 25   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {DEVICE}")

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
        img_name = os.path.join(self.root_dir, rel_path)
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            image = self.transform(image)
        labels = self.data.iloc[idx][LABELS].values.astype('float32')
        return image, torch.tensor(labels)

def get_pos_weights(dataset):
    df = dataset.data
    pos_weights = []
    for label in LABELS:
        n_pos = (df[label] == 1).sum()
        n_neg = (df[label] == 0).sum()
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
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / ACCUM_STEPS 

        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * ACCUM_STEPS
        loop.set_postfix(loss=loss.item() * ACCUM_STEPS)
        
    return running_loss / len(loader)

def validate_auc(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    aucs = []
    for i in range(len(LABELS)):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(score)
        except:
            pass 
            
    return np.mean(aucs)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    mean = [0.506, 0.506, 0.506]
    std =  [0.290, 0.290, 0.290]
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),       
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    print("Loading Data...")
    train_ds = ChexpertDataset(TRAIN_CSV, root_dir=IMAGE_ROOT, transform=train_transform)
    val_ds = ChexpertDataset(VAL_CSV, root_dir=IMAGE_ROOT, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print("Initializing Model (With MLP Head)...")
    model = CheXpertModel(num_classes=14, backbone_name='densenet121', pretrained=True)
    model = model.to(DEVICE)
    
    # REVERTED TO BCE LOSS (Safe + Stable)
    pos_weights = get_pos_weights(train_ds)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    scaler = torch.cuda.amp.GradScaler() 
    
    # --- PHASE 1: WARMUP ---
    print("\n--- PHASE 1: WARMUP (Frozen Backbone) ---")
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=EPOCHS_WARMUP)

    for epoch in range(EPOCHS_WARMUP):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, is_frozen=True)
        warmup_scheduler.step()
        print(f"Warmup Epoch {epoch+1}: Train Loss {train_loss:.4f}")
        
    # --- PHASE 2: FINE-TUNING ---
    print("\n--- PHASE 2: FINE-TUNING (Unfrozen Backbone) ---")
    for param in model.features.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW([
            {"params": model.features.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ], weight_decay=1e-2)
    
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE, eta_min=1e-6)
    
    best_mean_auc = 0.0 
    
    for epoch in range(EPOCHS_FINETUNE):
        display_epoch = epoch + EPOCHS_WARMUP
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, display_epoch, is_frozen=False)
        
        val_auc = validate_auc(model, val_loader)
        cosine_scheduler.step()
        
        print(f"Fine-tune Epoch {display_epoch+1}: Train Loss {train_loss:.4f} | Val AUC {val_auc:.4f}")
        
        if val_auc > best_mean_auc:
            best_mean_auc = val_auc
            save_path = os.path.join(SAVE_DIR, "chexpert_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Best Model Saved (AUC {best_mean_auc:.4f})")
            
    print("\nTraining Complete.")
