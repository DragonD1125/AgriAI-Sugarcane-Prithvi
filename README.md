# AgriAI-Prithvi: Multi-Crop Segmentation with Prithvi Foundation Model

Deep learning-based crop segmentation for Sugarcane and Rice detection in India using Sentinel-2 satellite imagery and IBM's Prithvi-100M geospatial foundation model.

## 🎯 Project Overview

This project fine-tunes the Prithvi-100M vision transformer for semantic segmentation of agricultural crops (Sugarcane and Rice) from multi-temporal Sentinel-2 imagery.

### Key Results

| Model | Task | Best IoU/F1 |
|-------|------|-------------|
| Prithvi Sugarcane | Binary | **77.1% F1** |
| Prithvi Rice | Binary | 37.5% IoU |
| Prithvi Multiclass | Sugar + Rice | 32% mIoU |
| U-Net Baseline | Binary | 54% F1 |

## 📁 Project Structure

```
Prithvi/
├── scripts/
│   ├── kaggle_prithvi_finetune.py      # Sugarcane fine-tuning (Kaggle)
│   ├── kaggle_prithvi_multiclass*.py   # Multiclass training scripts
│   ├── kaggle_prithvi_rice_only.py     # Rice-only binary training
│   ├── kaggle_unet_multiclass.py       # U-Net baseline
│   ├── extract_rice_coordinates*.py    # Rice coordinate extraction
│   ├── create_rice_dataset*.py         # Dataset creation pipeline
│   ├── merge_to_multiclass*.py         # Dataset merging
│   ├── evaluate_*.py                   # Model evaluation
│   └── visualize_rice_patches.py       # Data visualization
├── models/                             # Trained model weights (not in git)
├── data/                               # Training data (not in git)
└── outputs/                            # Visualizations and results
```

## 🛠️ Setup

### Requirements
```bash
pip install torch numpy rasterio pystac-client planetary-computer
pip install huggingface_hub timm einops
```

### For Kaggle Training
```bash
pip install segmentation-models-pytorch  # For U-Net baseline
```

## 📊 Data Pipeline

1. **Extract Coordinates** - From ground truth GeoTIFFs
2. **Download Sentinel-2** - Via Microsoft Planetary Computer
3. **Create Patches** - 224×224 multi-temporal patches
4. **Merge Dataset** - Combine crops into training NPZ

```python
# Example: Extract rice coordinates
python scripts/extract_rice_coordinates_all.py

# Create patches from Sentinel-2
python scripts/create_rice_dataset_v2.py

# Merge into training dataset
python scripts/merge_to_multiclass_v2.py
```

## 🚀 Training

Training is designed for **Kaggle GPUs (T4 x2)**:

1. Upload dataset NPZ to Kaggle
2. Copy training script to notebook
3. Run with GPU enabled

### Sugarcane (Best: 77% F1)
```python
# scripts/kaggle_prithvi_finetune.py
```

### Multiclass (Sugar + Rice)
```python
# scripts/kaggle_prithvi_multiclass_v3.py
```

## 📈 Model Architecture

Uses Prithvi-100M with modifications:
- **Encoder:** 6-12 layer Vision Transformer (pretrained)
- **Decoder:** Multi-stage upsampling CNN
- **Input:** 6 bands × 3 timestamps × 224×224 pixels
- **Output:** Semantic segmentation mask

## 📚 Data Sources

- **Sugarcane:** Di Tommaso et al. (2024) global dataset
- **Rice:** Classified rice maps (2018, 2020, 2022)
- **Imagery:** Sentinel-2 L2A via Microsoft Planetary Computer

## 📝 Key Findings

1. **Prithvi pretraining helps** - Outperforms from-scratch models
2. **Sugarcane is easier** - Distinct spectral signature (77% F1)
3. **Rice is challenging** - Similar to other vegetation (37% IoU)
4. **Multiclass adds complexity** - Performance drops when combining crops

## 📄 License

MIT License
