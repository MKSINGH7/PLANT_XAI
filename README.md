# PLANT_XAI

# Sugarcane Plant Disease Detection & Explainable AI (XAI)

An end-to-end Deep Learning and Explainable AI (XAI) pipeline designed to classify sugarcane plant diseases and evaluate interpretability methods. This project benchmarks modern Convolutional Neural Network architectures (**ConvNeXt-Tiny**, **EfficientNetV2-S**, **MobileNetV4**) on the *Sugarcane Plant Disease Dataset of Assam* and evaluates visual explanation techniques to ensure model predictions align with actual disease pathology.

## 📌 Project Overview

- **Objective 1:** Train and evaluate high-performing lightweight vision backbones for 9-class sugarcane disease classification.
- **Objective 2:** Generate visual explanations using gradient-based and activation-based XAI methods (**Grad-CAM++**, **Layer-CAM**, **Eigen-CAM**).
- **Objective 3 & 4 (In Progress):** Perform quantitative evaluation of visual heatmaps using Deletion/Insertion AUC metrics and the Pointing Game.

## 📊 Dataset Structure & Statistics

The dataset consists of **5,769 high-resolution images** categorized into 9 distinct classes (8 disease types + 1 healthy control):

| **Class Name**          | **Image Count** | **Description / Symptoms**                                   |
| ----------------------- | --------------- | ------------------------------------------------------------ |
| **Grassy Shoot**        | 611             | Excessive tillering with thin, chlorotic shoots              |
| **Healthy**             | 679             | Normal green foliage without visible lesions                 |
| **Mosaic**              | 611             | Discolored, mottled green/yellow patches on leaves           |
| **Pokkah Boeng**        | 611             | Distorted, malformed, or twisted top leaves                  |
| **Red Leaf Spot**       | 611             | Small reddish-brown spots expanding across foliage           |
| **Red Rot**             | 678             | Reddish lesions with intermingled white spots                |
| **Ring Spot**           | 679             | Elliptical spots with straw-colored centers and dark margins |
| **Wilt**                | 611             | Stunted growth, foliage yellowing, and drying                |
| **Yellow Leaf Disease** | 678             | Yellowing of the midrib expanding toward leaf margins        |

### Dataset Management Highlights

- **Automated Extraction & Discovery:** Safely extracts datasets and dynamically discovers class structures.
- **Data Validation:** Verifies image integrity via MD5 hashing and PIL validation.
- **Leakage-Free Splits:** Uses **Stratified Group $K$-Fold** splitting to group near-duplicate/augmented copies and eliminate data leakage between `train`, `val`, and `test` splits.

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Framework:** PyTorch (with Mixed Precision / CUDA support)
- **Model Library:** `timm` (PyTorch Image Models)
- **Augmentation:** `albumentations`
- **XAI Library:** `grad-cam`
- **Evaluation & Utilities:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `opencv-python-headless`, `tqdm`

2. Configuration

Set up your project paths and training hyperparameters inside CONFIG:

Python
CONFIG = {
    "ZIP_PATH": "/content/Sugarcane Plant Disease Dataset of Assam.zip",
    "OUTPUT_DIR": "/content/sugarcane_xai_outputs",
    "IMAGE_SIZE": 224,
    "SEED": 42,
    "MODELS": ["convnext_tiny", "efficientnetv2_s", "mobilenetv4"],
    "MOBILENETV4_VARIANT": "mobilenetv4_conv_medium",
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 20,
    "LEARNING_RATE": 1e-4,
    "USE_AMP": True,
    "XAI_METHODS": ["gradcam++", "layercam", "eigencam"],
}

3. Training & Evaluation

Run the notebook or main script (sugarcane_xai_colab.ipynb) sequentially to:

Extract and validate images.
Generate group-stratified splits.
Train selected backbones with automatic checkpointing (checkpoint.pt).
Generate comparative XAI heatmaps for test samples.
💡 Explainable AI (XAI) Methods

This pipeline incorporates three visual explanation methods to assess model focus areas:

Grad-CAM++: Enhances pixel-level attribution over standard Grad-CAM, suitable for instances with multiple occurrences of a disease symptom.
Layer-CAM: Produces fine-grained heatmaps by combining activations and gradients from deep layers.
Eigen-CAM: An activation-based, gradient-free approach using principal components to highlight salient regions.
📁 Output Directory Hierarchy

Plaintext

sugarcane_xai_outputs/
├── extracted/          # Unzipped dataset
├── checkpoints/        # Saved model weights & training states (.pth / .pt)
├── plots/              # Training/Validation loss & accuracy curves
├── results/            # Confusion matrices & classification reports (CSV/JSON)
└── xai/                # Generated visual explanation heatmap
