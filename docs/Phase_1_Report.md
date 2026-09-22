# RailGuard Vision — Technical Report: Phase 1 (Dataset Setup & Preparation)

## 1. Executive Summary
Phase 1 establishes the dataset acquisition, cleaning, stratified splitting, annotation workflow, and data augmentation pipeline for the **RailGuard Vision** railway track defect detection system.

## 2. Methodology & Implementation Steps

### 2.1 Kaggle Dataset Download & Inspection
Using the `kaggle` Python API, the "Railway Track Fault Detection" dataset was fetched into `./data/raw/` and unzipped. An automated directory scanner evaluated class balance across surface defect categories (`crack`, `missing_fastener`, `broken_rail`, `joint_fault`).

### 2.2 Data Cleaning & Standardizing
- **Corruption Purging**: Scanned images using PIL `verify()` and OpenCV `imread()`. Corrupt or truncated files were logged and deleted.
- **MD5 Exact Duplicate Removal**: Computed 128-bit MD5 hashes of raw image byte streams to detect and eliminate duplicate images.
- **JPEG Standardization**: Resaved all valid images as 24-bit RGB JPEG files at 95% quality.

### 2.3 Stratified Train / Val / Test Splitting
Using `scikit-learn`'s `train_test_split`, images were stratified by defect class to ensure identical class proportions across:
- **Train Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

### 2.4 Roboflow Bounding-Box Annotation & `data.yaml`
Public classification datasets lack spatial bounding box coordinates `[x_center, y_center, width, height]`. Images were exported to Roboflow for bounding box polygon labeling. A YOLO `data.yaml` configuration file was generated establishing class IDs:
```yaml
path: ./data/dataset_split
train: train/images
val: val/images
test: test/images

names:
  0: crack
  1: missing_fastener
  2: broken_rail
  3: joint_fault
```

### 2.5 Albumentations Data Augmentation
To enhance generalization under variable ambient outdoor railway conditions, an Albumentations pipeline was configured with:
- **Horizontal & Vertical Flips** ($p=0.5$)
- **Random 90° Rotations** ($p=0.5$)
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)** ($clip\_limit=3.0$, $p=0.8$)
- **Random Brightness & Contrast Adjustment** ($\pm 20\%$, $p=0.7$)

## 3. Results & Verification
- Cleaned dataset purges 0 corrupt images and eliminates duplicate records.
- Stratified 70/15/15 train/val/test splits maintain uniform class proportions.
- Generated notebook: `notebooks/Phase1_Dataset_Preparation.ipynb`.
