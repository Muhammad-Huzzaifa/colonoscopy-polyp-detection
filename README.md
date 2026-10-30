# Colonoscopy Polyp Detection: Classical Enhancement & Deep Learning CADe Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Modal GPU](https://img.shields.io/badge/Compute-Modal%20Serverless%20GPU-7FEE64)](https://modal.com/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-FFD21E)](https://huggingface.co/Muhammad-Huzzaifa/colonoscopy-polyp-detection)
[![Dataset: Kvasir-SEG](https://img.shields.io/badge/Dataset-Kvasir--SEG-blue)](https://datasets.simula.no/kvasir-seg/)

An end-to-end, configuration-driven Computer-Aided Detection (**CADe**) framework for real-time localization of colorectal polyps in optical colonoscopy frames. This repository systematically benchmarks how classical spatial and frequency/luminance preprocessing techniques impact single-stage anchor-free object detectors (`YOLOv8m`) on the **Kvasir-SEG** dataset.

> **Lightweight Repo Architecture:** All experimental metrics, training curves, confusion matrices, and qualitative evaluations are version-controlled inside [`results/`](./results), while heavy model checkpoints (`best.pt`) are automatically pushed to the [**Hugging Face Model Hub**](https://huggingface.co/Muhammad-Huzzaifa/colonoscopy-polyp-detection).

---

## Key Highlights & Experimental Scope

Colorectal cancer (CRC) predominantly arises from adenomatous polyps, yet clinical colonoscopy suffers from a **15%–30% miss rate** due to uneven single-point LED illumination, mucosal fold shadows, wet specular glare, and faint boundaries on flat/sessile lesions. 

This repository evaluates three controlled experimental pipelines on an identical `80% / 10% / 10%` (`seed=42`) split of **Kvasir-SEG**:

1. **Experiment 1 — Baseline (`exp1_baseline`):** Reproduction of the base paper architecture (**YOLOv8m** with `CSPDarknet53` + `C2f` backbone, `PAN-FPN` neck, and decoupled `CIoU + DFL` head) trained on raw RGB colonoscopy frames.
2. **Experiment 2 — Luminance Contrast Normalization (`exp2_preproc1`):** Applies $3 \times 3$ **Gaussian Low-Pass Smoothing** ($\sigma = 0.5$) followed by **Contrast Limited Adaptive Histogram Equalization (CLAHE)** strictly on the Luminance ($L$) channel in perceptual **CIE LAB** space (`clipLimit=2.0`, `tileGridSize=8x8`) to illuminate dark haustral folds without distorting vascular chrominance ($a, b$).
3. **Experiment 3 — Boundary & Edge Sharpening (`exp3_preproc2`):** Combines $5 \times 5$ **Gaussian Unsharp Masking** ($k = 1.2$, $\sigma = 1.2$) with second-order **Laplacian High-Boost Filtering** ($\nabla^2 L$, $w = 0.15$) to accentuate subtle mucosal elevation contours of flat and sessile polyps.

![Preprocessing Visual Comparison](results/dataset_stats/preprocessing_comparison_samples.png)

---

## Benchmark Results (Held-Out Test Split)

All models were trained for 80 epochs (`batch_size=16`, `imgsz=640`, `AdamW`, Cosine Annealing LR) on an **NVIDIA A10G (24 GB VRAM)** GPU via Modal Serverless infrastructure.

| Metric | Exp 1: Baseline (Raw RGB) | Exp 2: LAB-CLAHE + Gaussian | Exp 3: Unsharp + Laplacian | Best Delta vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 88.29% | **92.28%** | 89.69% | **+3.99%** (Exp 2) |
| **Recall** | **91.35%** | 88.46% | 90.38% | Baseline |
| **F1-Score** | 89.79% | **90.33%** | 90.04% | **+0.54%** (Exp 2) |
| **mAP@0.50** | 92.63% | **92.69%** | 92.68% | **+0.06%** (Exp 2) |
| **mAP@0.50:0.95** | 74.57% | **75.67%** | 75.02% | **+1.10%** (Exp 2) |
| **Inference Latency** | 7.56 ms (~132 FPS) | **6.16 ms (~162 FPS)** | 6.44 ms (~155 FPS) | **-1.40 ms** (Exp 2) |
| **Pretrained Weights** | [HF Hub (`exp1`)](https://huggingface.co/Muhammad-Huzzaifa/colonoscopy-polyp-detection/tree/main/exp1_baseline) | [HF Hub (`exp2`)](https://huggingface.co/Muhammad-Huzzaifa/colonoscopy-polyp-detection/tree/main/exp2_preproc1) | [HF Hub (`exp3`)](https://huggingface.co/Muhammad-Huzzaifa/colonoscopy-polyp-detection/tree/main/exp3_preproc2) | — |

![Cross-Experiment Metrics Chart](results/comparison/comparison_metrics_chart.png)

### Key Findings
* **Precision & Localization Boost via LAB-CLAHE:** Decoupling luminance in CIE LAB space and normalizing local contrast reduced false-positive detections on shadowed folds, boosting **Precision by +3.99%** ($88.29\% \rightarrow 92.28\%$) and strict multi-IoU bounding-box tightness (**mAP@0.50:0.95**) by **+1.10%** ($74.57\% \rightarrow 75.67\%$).
* **Balanced Edge Enhancement:** Spatial Unsharp Masking + Laplacian filtering achieved the most balanced Precision/Recall profile (**89.69% Precision**, **90.38% Recall**, **90.04% F1**) by sharpening faint sessile polyp boundaries.
* **Qualitative Performance:**

| Exp 1: Baseline Predictions | Exp 2: LAB-CLAHE Predictions | Exp 3: Unsharp-Laplacian Predictions |
| :---: | :---: | :---: |
| ![Exp 1 Preds](results/exp1_baseline/qualitative_predictions.png) | ![Exp 2 Preds](results/exp2_preproc1/qualitative_predictions.png) | ![Exp 3 Preds](results/exp3_preproc2/qualitative_predictions.png) |

---

## Repository Structure

```text
colonoscopy-polyp-detection/
├── configs/
│   ├── exp1_baseline.yaml
│   ├── exp2_preproc1.yaml
│   └── exp3_preproc2.yaml
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   └── hf_utils.py
├── results/
│   ├── dataset_stats/
│   ├── exp1_baseline/
│   ├── exp2_preproc1/
│   ├── exp3_preproc2/
│   └── comparison/
├── train.py
├── train_modal.py
├── Makefile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Dataset Overview (Kvasir-SEG)

The pipeline automatically downloads and formats the official Kvasir-SEG dataset from SimulaMet:

* **Total Images:** 1,000 RGB colonoscopy frames ($332 \times 487$ to $1920 \times 1072$ native resolution, letterbox-resized to $640 \times 640$).
* **Total Annotations:** 1,071 polyp bounding boxes across 945 single-polyp and 55 multi-polyp frames (`0: polyp`).
* **Deterministic Split (`seed=42`):** 800 Train / 100 Validation / 100 Test images.

### Quickstart & Usage

#### 1. Installation

```bash
git clone https://github.com/Muhammad-Huzzaifa/colonoscopy-polyp-detection.git
cd colonoscopy-polyp-detection
python3 -m venv .venv && source .venv/bin/activate
make install
```

Configure your Hugging Face write token (optional, required only if pushing new weights to the Hub):

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
```

#### 2. Prepare Dataset & Classical Preprocessed Views

Downloads Kvasir-SEG, converts `kvasir_bboxes.json` into normalized YOLO labels, builds the 80/10/10 split, and generates cached image views for all preprocessing pipelines:

```bash
make data
# Or run directly: python -m src.prepare_data --all
```

#### 3. Train & Evaluate (Local GPU/CPU or Modal Serverless GPU)

##### Option A — Run on Modal Cloud GPU (Recommended):

```bash
make modal-exp1      # Experiment 1: Baseline (Raw)
make modal-exp2      # Experiment 2: LAB-CLAHE + Gaussian
make modal-exp3      # Experiment 3: Unsharp Masking + Laplacian
```

##### Option B — Run Locally:

```bash
make local-exp1
make local-exp2
make local-exp3
```

#### 4. Generate Cross-Experiment Benchmark Table & Charts

```bash
make compare
# Outputs results/comparison/comparison_table.md and comparison_metrics_chart.png
```

### Citation & Credits

* **Base Paper:** Lalinia, M., & Sahafi, A. (2024). Colorectal polyp detection in colonoscopy images using YOLO-V8 network. *Signal, Image and Video Processing*, 18(3), 2047–2058.
* **Dataset:** Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. MultiMedia Modeling (MMM 2020).
* **Contributors:** Muhammad Huzaifa (`23F-0043`) & Talha Zahid (`23F-0004`) — Department of Artificial Intelligence & Data Science, FAST-NUCES.
