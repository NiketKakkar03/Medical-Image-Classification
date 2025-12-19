import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'data/'
RAW_CSV_PATH = os.path.join(DATA_DIR, "cheXpert-v1.0-small/train.csv")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "cheXpert-v1.0-small/train_processed.csv")
TRAIN_OUTPUT = os.path.join(DATA_DIR, "cheXpert-v1.0-small/train_split.csv")
VAL_OUTPUT = os.path.join(DATA_DIR, "cheXpert-v1.0-small/val_split.csv")
IMG_SIZE = 224

# The 14 target pathologies
LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

def preprocess_labels(df):
    print(f"Original Row Count: {len(df)}")
    
    # 1. FILTER: Keep only Frontal images
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal'].copy()
        print(f"Row Count after Frontal Filter: {len(df)}")
    
    # 2. FILL NA: Treat blanks as 0.0 (Negative)
    df[LABELS] = df[LABELS].fillna(0.0)
    
    # 3. POLICY: U-Ones Strategy
    # Convert -1 (Uncertain) to 1 (Positive)
    # This maximizes sensitivity for Atelectasis/Edema/Consolidation
    df[LABELS] = df[LABELS].replace(-1.0, 1.0)
    
    # 4. METADATA: Handle AP/PA
    # Fill missing views with 'Unknown' to avoid errors later
    df['AP/PA'] = df['AP/PA'].fillna('Unknown')
    
    return df

def check_images_and_calc_stats(df, root_dir):
    """
    1. Verifies image existence.
    2. Calculates Mean/Std for the dataset.
    3. Returns valid rows only.
    """
    valid_indices = []
    
    # Accumulators for Mean/Std calculation (Welford's method or simple sum)
    # Using simple sum for readability on large datasets
    # Note: Running this on 200k images takes time. 
    # We will run on a sample of 5000 images for speed to get an estimate.
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    count = 0
    
    print("Verifying images and calculating statistics (Sampled)...")
    
    # sample for stats calculation to save time
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = row['Path']
        # ADJUST PATH IF NEEDED:
        # rel_path = rel_path.replace("CheXpert-v1.0-small/", "")
        
        full_path = os.path.join(root_dir, rel_path)
        
        if os.path.exists(full_path):
            valid_indices.append(idx)
            
            # If this row is in our 'stats sample', read it and calculate
            if idx in sample_df.index:
                try:
                    # Read BGR (OpenCV default) -> Convert to RGB -> Normalize 0-1
                    img = cv2.imread(full_path)
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    
                    pixel_sum += img.sum(axis=(0, 1))
                    pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
                    count += (img.shape[0] * img.shape[1])
                except:
                    pass
        else:
            # You might want to log missing files
            pass

    # Filter DataFrame to keep only existing images
    df_clean = df.loc[valid_indices].reset_index(drop=True)
    
    # Finalize Stats
    mean = pixel_sum / count
    std = np.sqrt((pixel_sq_sum / count) - (mean ** 2))
    
    print(f"\nFinal Valid Image Count: {len(df_clean)}")
    print(f"Computed Mean: {mean}")
    print(f"Computed Std:  {std}")
    
    return df_clean, mean, std

def extract_patient_id(path):
    parts = path.split('/')
    for part in parts:
        if "patient" in part:
            return part
    return "unknown"


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Raw CSV
    # print("Loading CSV...")
    # df = pd.read_csv(RAW_CSV_PATH)
    
    # # 2. Process Labels
    # df_processed = preprocess_labels(df)
    
    # # 3. Validate Images & Get Stats
    # df_final, mean, std = check_images_and_calc_stats(df_processed, DATA_DIR)
    
    # # 4. Save
    # df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    # print(f"\nSUCCESS! Processed CSV saved to: {OUTPUT_CSV_PATH}")
    # print("Use this CSV for all future model training.")
    
    # # Save stats to a text file for reference
    # with open(os.path.join(DATA_DIR, "dataset_stats.txt"), "w") as f:
    #     f.write(f"Mean: {mean}\n")
    #     f.write(f"Std: {std}\n")

    print("Loading Processed CSV...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    df['PatientID'] = df['Path'].apply(extract_patient_id)

    num_patients = df['PatientID'].nunique()
    print(f"Total Images: {len(df)}")
    print(f"Total Unique Patients: {num_patients}")

    splitter = GroupShuffleSplit(test_size=0.10, n_splits=1, random_state=42)
    
    # We only need the indices
    train_idx, val_idx = next(splitter.split(df, groups=df['PatientID']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print("\nSplit Results:")
    print(f"Training Set:   {len(train_df)} images ({train_df['PatientID'].nunique()} patients)")
    print(f"Validation Set: {len(val_df)} images ({val_df['PatientID'].nunique()} patients)")
    
    # 3. Validation Check (Leakage Test)
    train_patients = set(train_df['PatientID'].unique())
    val_patients = set(val_df['PatientID'].unique())
    
    overlap = train_patients.intersection(val_patients)
    if len(overlap) == 0:
        print("\n No patient overlap detected (Zero Leakage).")
    else:
        print(f"\n {len(overlap)} patients are in both sets!")
        exit()

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    val_df.to_csv(VAL_OUTPUT, index=False)
    print(f"\nSaved to:\n  {TRAIN_OUTPUT}\n  {VAL_OUTPUT}")