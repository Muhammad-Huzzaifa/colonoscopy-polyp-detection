| Hyperparameter / Setting | Value |
| :--- | :--- |
| Model Architecture | yolov8m.pt |
| Input Image Size | 640x640 |
| Batch Size | 16 |
| Number of Epochs | 80 |
| Learning Rate (lr0 / lrf) | 0.001 (final: 0.01) |
| Optimizer | AdamW |
| Loss Function | CIoU (Box Regression) + DFL (Distribution Focal Loss) + BCE (Classification) |
| Scheduler | Cosine Annealing |
| Train/Val/Test Split | 80% / 10% / 10% (800 / 100 / 100) |
| Data Augmentation | Mosaic (1.0), Horizontal Flip (0.5), HSV-Hue (0.015), HSV-Sat (0.7), HSV-Val (0.4), Scale (0.5) |
| Classical Preprocessing | 3x3 Gaussian smoothing followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Luminance (L) channel in CIE LAB space. |
