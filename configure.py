import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
path = 'data/cheXpert-v1.0-small/train.csv'
df = pd.read_csv(path)

# 2. Filter for Frontal Images ONLY (Standard practice)
# This simplifies the problem significantly.
df_frontal = df[df['Frontal/Lateral'] == 'Frontal'].copy()
print(f"Original size: {len(df)}, Frontal only: {len(df_frontal)}")

# 3. Analyze AP/PA Blanks
missing_appa = df_frontal['AP/PA'].isnull().sum()
print(f"Missing AP/PA values in Frontal images: {missing_appa}")

# Fill missing AP/PA with 'Unknown' so we can visualize them
df_frontal['AP/PA'] = df_frontal['AP/PA'].fillna('Unknown')

# 4. Analyze All 14 Pathologies
all_labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# Create a summary dataframe for visualization
label_counts = []

for label in all_labels:
    # Get counts of 1.0 (Positive), 0.0 (Negative), -1.0 (Uncertain), and NaN
    counts = df_frontal[label].value_counts(dropna=False)
    
    label_counts.append({
        'Label': label,
        'Positive': counts.get(1.0, 0),
        'Uncertain': counts.get(-1.0, 0),
        'Negative': counts.get(0.0, 0) + counts.get(float('nan'), 0) # Treat NaN as Negative
    })

df_counts = pd.DataFrame(label_counts).set_index('Label')

# 5. Visualize
plt.figure(figsize=(12, 8))
# Normalize to percentages so we can compare prevalence easily
df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100

df_percent.plot(kind='barh', stacked=True, color=['green', 'orange', 'lightgray'], ax=plt.gca())
plt.title('Distribution of All 14 Pathologies (Frontal Images)')
plt.xlabel('Percentage')
plt.legend(title='Class', loc='upper right', labels=['Positive (1)', 'Uncertain (-1)', 'Negative (0)'])
plt.tight_layout()
plt.show()

# 6. Check Correlation (Heatmap)
# Convert -1 to 1 (U-Ones policy) for correlation check to see "Potential Positives"
df_numeric = df_frontal[all_labels].replace(-1.0, 1.0).fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Pathologies')
plt.show()