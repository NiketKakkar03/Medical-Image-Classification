import pandas as pd
import os

path = 'data/cheXpert-v1.0-small/train.csv'
df = pd.read_csv(path)

# print(df.head())

first_image_rel_path = df.iloc[0]['Path']
full_path = os.path.join('data', first_image_rel_path)

if os.path.exists(full_path):
    print(f"SUCCESS: Found image at {full_path}")
else:
    print(f"ERROR: Could not find image at {full_path}")
    print("Check your folder naming!")