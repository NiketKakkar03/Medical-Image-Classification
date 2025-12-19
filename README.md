# CheXpert Medical Image Classification (PyTorch)

A modular Deep Learning pipeline for multi-label classification of chest X-rays using the **CheXpert** dataset. This project implements a custom **DenseNet121** architecture with **GeM Pooling** and is optimized for training on memory-constrained hardware (e.g., RTX 3050 Laptop with 4GB VRAM).

## 📌 Project Overview
*   **Task:** Multi-Label Binary Classification (14 pathologies).
*   **Dataset:** CheXpert-v1.0-small (Frontal Views Only).
*   **Model:** DenseNet121 Backbone + GeM Pooling Neck + Linear Head.
*   **Training Strategy:** Hybrid (Frozen Warmup -> Unfrozen Fine-tuning) with Mixed Precision (AMP).
*   **Performance:** Average AUC **0.7714** (Best Class: No Finding 0.8744).

## 🏗️ Architecture
The model uses a modular design (`model_architecture.py`) to allow for future expansion (e.g., Multi-View Fusion).

1.  **Backbone:** `DenseNet121` (Pretrained on ImageNet). Selected for its feature reuse and efficiency.
2.  **Neck:** **GeM (Generalized Mean) Pooling**. A learnable pooling layer that captures fine-grained features better than standard Global Average Pooling.
3.  **Head:** Dropout (`p=0.2`) -> Linear Layer (1024 -> 14 Classes).
4.  **Optimization:**
    *   **Loss:** `BCEWithLogitsLoss` with **Positive Class Weighting** to handle imbalance (e.g., rare Fractures).
    *   **Optimizer:** `AdamW` (Weight Decay for regularization).
    *   **Hardware:** Gradient Accumulation + Automatic Mixed Precision (AMP) to enable Batch Size 32 equivalent on 4GB VRAM.

## 📊 Dataset & Preprocessing
*   **Filtering:** Restricted to **Frontal** views (AP/PA) only to reduce anatomical confusion.
*   **Policy:** Implemented **"U-Ones"** strategy: Uncertain labels (`-1`) are mapped to Positive (`1`) to maximize model sensitivity.
*   **Normalization:** Custom calculated statistics for this dataset:
    *   Mean: `[0.506, 0.506, 0.506]`
    *   Std: `[0.290, 0.290, 0.290]`
*   **Splitting:** **GroupShuffleSplit** used to ensure zero patient overlap between Train and Validation sets (prevents data leakage).

## 🚀 Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/yourusername/chexpert-classifier.git
    cd chexpert-classifier
    ```

2.  **Install Dependencies:**
    *   *Note for RTX 3050 Users:* Ensure you install the CUDA-enabled version of PyTorch.
    ```
    # Remove CPU version if present
    pip uninstall torch torchvision torchaudio

    # Install CUDA 12.x version
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    
    # Install other requirements
    pip install pandas numpy opencv-python tqdm scikit-learn pillow
    ```

## 🖥️ Usage

### 1. Preprocessing
Converts raw data into a clean CSV with U-Ones policy and calculates stats.\


### 2. Create Splits
Generates patient-level Train/Val splits to prevent leakage.



### 3. Training
Runs the 5-epoch hybrid training loop (Warmup + Fine-tuning) and saves the best model to `models/chexpert_best.pth`.

### 4. Evaluation
Generates the AUC Report Card for all 14 diseases.


## 📈 Performance Results
**Validation Set (Epoch 5)**

| Pathology | AUC Score | Status |
| :--- | :--- | :--- |
| **No Finding** | **0.8744** | 🟢 Excellent |
| **Cardiomegaly** | **0.8504** | 🟢 Excellent |
| **Pleural Effusion** | **0.8484** | 🟢 Excellent |
| **Edema** | **0.8302** | 🟢 Excellent |
| **Support Devices** | **0.8242** | 🟢 Excellent |
| **Pneumothorax** | **0.8145** | 🟢 Excellent |
| Pleural Other | 0.8008 | 🟢 Good |
| Lung Lesion | 0.7632 | 🟡 Fair |
| Fracture | 0.7506 | 🟡 Fair (High for rare class) |
| Pneumonia | 0.7324 | 🟡 Fair |
| Lung Opacity | 0.7127 | 🟡 Fair |
| Atelectasis | 0.6914 | 🔴 Challenging |
| Consolidation | 0.6736 | 🔴 Challenging |
| Enlarged Cardiomediastinum | 0.6324 | 🔴 Challenging |
| **AVERAGE** | **0.7714** | |

## 🔮 Future Improvements
*   **Lateral View Fusion:** Integrate lateral X-rays using a multi-view backbone to improve performance on "Enlarged Cardiomediastinum" and "Consolidation".
*   **Grad-CAM Integration:** Add explainability visualization to the inference pipeline.
*   **Hyperparameter Tuning:** Experiment with `EfficientNet-B0` for potential speed gains.
