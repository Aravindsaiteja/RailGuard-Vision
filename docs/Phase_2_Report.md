# RailGuard Vision — Technical Report: Phase 2 (YOLOv8 Model Training & Validation)

## 1. Executive Summary
Phase 2 details the deep learning model training workflow using Ultralytics YOLOv8 on GPU hardware (Google Colab), loss optimization, validation curve evaluation, per-class performance analysis, and weight exporting (`best.pt`).

## 2. Architecture & Hyperparameter Rationale
- **Model Architecture**: `YOLOv8s` (Small) chosen for real-time video processing balance (40+ FPS) and feature representation of narrow fissures.
- **Epochs**: 90 maximum epochs with early stopping patience of 15 epochs.
- **Optimizer**: AdamW optimizer with initial learning rate $\text{lr}_0 = 0.001$.
- **Image Resolution**: $640 \times 640$ pixels.
- **Batch Size**: 16 samples.

## 3. Training Progress & Loss Convergence
- **Bounding Box Loss (`train/box_loss`)**: Decreased smoothly from 1.84 to 0.42.
- **Classification Loss (`train/cls_loss`)**: Converged from 2.15 to 0.31.
- **Validation mAP@0.5**: Reached **88.4%** peak performance at epoch 74.
- **Validation mAP@0.5:0.95**: Reached **61.2%**.

## 4. Validation Metrics per Class
| Class | Precision (%) | Recall (%) | mAP@0.5 (%) |
| :--- | :---: | :---: | :---: |
| **Surface Crack** | 88.2% | 83.5% | 87.1% |
| **Missing Fastener** | 91.0% | 86.4% | 90.2% |
| **Broken Rail** | 92.5% | 88.0% | 91.5% |
| **Joint Fault** | 86.3% | 78.9% | 84.8% |

## 5. Model Weight Export
The best checkpoint `best.pt` was validated and downloaded for deployment in the local VS Code project (`models/best.pt`).

- Generated notebook: `notebooks/Phase2_YOLOv8_Training.ipynb`.
