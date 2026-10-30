# src/evaluate.py
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate_and_export_experiment(
    model: Any,
    dataset_yaml: Path,
    config: Dict[str, Any],
    run_dir: Path,
    results_root: Path,
) -> Dict[str, Any]:
    """
    evaluates model performance on the held-out test split, exports loss curves, PR curves, confusion matrix, and qualitative predictions grid, and saves metrics to a JSON file

    Args:
        model (Any): an instance of the trained model (e.g., YOLO or RT-DETR)
        dataset_yaml (Path): path to the dataset YAML file specifying the test split
        config (Dict[str, Any]): configuration dictionary containing experiment and model parameters
        run_dir (Path): path to the directory where the model's training run artifacts are stored (e.g., loss curves, PR curves)
        results_root (Path): path to the root directory where the evaluation results will be saved

    Returns:
        Dict[str, Any]: a dictionary containing evaluation metrics such as precision, recall, F1-score, mAP@0.50, mAP@0.50:0.95, and inference speed
    """
    exp_id = config.get("experiment_id", "exp")
    exp_dir = results_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Evaluate] Running test-set evaluation for {exp_id} ...")
    val_results = model.val(
        data=str(dataset_yaml.resolve()),
        split="test",
        imgsz=config.get("model", {}).get("imgsz", 640),
        plots=True,
        save_json=False,
        verbose=False,
    )

    precision = float(val_results.box.p[0]) if len(val_results.box.p) > 0 else 0.0
    recall = float(val_results.box.r[0]) if len(val_results.box.r) > 0 else 0.0
    f1 = float(val_results.box.f1[0]) if len(val_results.box.f1) > 0 else 0.0
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)

    metrics_data = {
        "experiment_id": exp_id,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "inference_speed_ms": {
            "preprocess": round(float(val_results.speed.get("preprocess", 0.0)), 2),
            "inference": round(float(val_results.speed.get("inference", 0.0)), 2),
            "postprocess": round(float(val_results.speed.get("postprocess", 0.0)), 2),
        },
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    _export_hyperparameter_table(config, exp_dir)
    _harvest_yolo_plots(run_dir, exp_dir)
    _generate_qualitative_predictions_grid(
        model=model,
        dataset_yaml=dataset_yaml,
        output_path=exp_dir / "qualitative_predictions.png",
        num_samples=8,
    )

    print(f"[Evaluate] Artifacts saved to: {exp_dir}")
    return metrics_data


def _export_hyperparameter_table(config: Dict[str, Any], exp_dir: Path) -> None:
    """
    exports the model's hyperparameters to a JSON file and a Markdown table

    Args:
        config (Dict[str, Any]): configuration dictionary containing experiment and model parameters
        exp_dir (Path): path to the directory where the hyperparameter files will be saved
    """
    t_cfg = config.get("training", {})
    m_cfg = config.get("model", {})
    p_cfg = config.get("preprocessing", {})

    settings_table = [
        {"Hyperparameter / Setting": "Model Architecture", "Value": str(m_cfg.get("architecture"))},
        {"Hyperparameter / Setting": "Input Image Size", "Value": f"{m_cfg.get('imgsz', 640)}x{m_cfg.get('imgsz', 640)}"},
        {"Hyperparameter / Setting": "Batch Size", "Value": str(t_cfg.get("batch_size"))},
        {"Hyperparameter / Setting": "Number of Epochs", "Value": str(t_cfg.get("epochs"))},
        {"Hyperparameter / Setting": "Learning Rate (lr0 / lrf)", "Value": f"{t_cfg.get('lr0')} (final: {t_cfg.get('lrf')})"},
        {"Hyperparameter / Setting": "Optimizer", "Value": str(t_cfg.get("optimizer"))},
        {"Hyperparameter / Setting": "Loss Function", "Value": str(t_cfg.get("loss_summary"))},
        {"Hyperparameter / Setting": "Scheduler", "Value": "Cosine Annealing" if t_cfg.get("cos_lr") else "Linear"},
        {"Hyperparameter / Setting": "Train/Val/Test Split", "Value": "80% / 10% / 10% (800 / 100 / 100)"},
        {"Hyperparameter / Setting": "Data Augmentation", "Value": str(t_cfg.get("augmentation_summary"))},
        {"Hyperparameter / Setting": "Classical Preprocessing", "Value": str(p_cfg.get("description"))},
    ]

    with open(exp_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(settings_table, f, indent=2)

    md_lines = ["| Hyperparameter / Setting | Value |", "| :--- | :--- |"]
    for row in settings_table:
        md_lines.append(f"| {row['Hyperparameter / Setting']} | {row['Value']} |")
    with open(exp_dir / "hyperparameters.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def _harvest_yolo_plots(run_dir: Path, exp_dir: Path) -> None:
    """
    copies the relevant YOLO-generated plots (loss curves, PR curve, confusion matrix, F1 curve) from the training run directory to the experiment results directory

    Args:
        run_dir (Path): path to the directory where the model's training run artifacts are stored
        exp_dir (Path): path to the directory where the experiment results will be saved
    """
    plot_mappings = {
        "results.png": "loss_curves.png",
        "PR_curve.png": "pr_curve.png",
        "confusion_matrix_normalized.png": "confusion_matrix.png",
        "F1_curve.png": "f1_curve.png",
    }
    for src_name, dst_name in plot_mappings.items():
        src_file = run_dir / src_name
        if src_file.exists():
            shutil.copy2(src_file, exp_dir / dst_name)


def _generate_qualitative_predictions_grid(
    model: Any, dataset_yaml: Path, output_path: Path, num_samples: int = 8
) -> None:
    """
    generates a grid of qualitative predictions for a given model and dataset

    Args:
        model (Any): an instance of the trained model (e.g., YOLO or RT-DETR)
        dataset_yaml (Path): path to the dataset YAML file specifying the test split
        output_path (Path): path where the qualitative predictions grid image will be saved
        num_samples (int, optional): number of samples to include in the grid. Defaults to 8.
    """
    dataset_dir = dataset_yaml.parent
    test_img_dir = dataset_dir / "images" / "test"
    test_lbl_dir = dataset_dir / "labels" / "test"

    img_paths = sorted(list(test_img_dir.glob("*.jpg")))[:num_samples]
    if not img_paths:
        return

    cols = 4
    rows = (len(img_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, img_p in enumerate(img_paths):
        ax = axes[idx]
        img_bgr = cv2.imread(str(img_p))
        h, w = img_bgr.shape[:2]
        canvas = img_bgr.copy()

        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = int((xc - bw / 2.0) * w)
                        y1 = int((yc - bh / 2.0) * h)
                        x2 = int((xc + bw / 2.0) * w)
                        y2 = int((yc + bh / 2.0) * h)
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(canvas, "GT", (x1, max(20, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        preds = model.predict(source=str(img_p), conf=0.25, verbose=False)[0]
        if preds.boxes is not None:
            for box in preds.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(canvas, f"Pred {conf:.2f}", (xyxy[0], xyxy[3] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(img_p.stem, fontsize=9)
        ax.axis("off")

    for idx in range(len(img_paths), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Ground Truth (Green) vs. Predictions (Red)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compile_benchmark_comparison(results_root: Path) -> None:
    """
    compiles a benchmark comparison of detection performance metrics across multiple experimental pipelines and generates a summary table and bar chart

    Args:
        results_root (Path): path to the root directory where the evaluation results of different experiments are stored
    """
    comp_dir = results_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Baseline (Raw)", results_root / "exp1_baseline" / "metrics.json"),
        ("Preprocessing 1 (LAB-CLAHE)", results_root / "exp2_preproc1" / "metrics.json"),
        ("Preprocessing 2 (Unsharp Edge)", results_root / "exp3_preproc2" / "metrics.json"),
    ]

    records: Dict[str, Dict[str, Any]] = {}
    for label, path in experiments:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                records[label] = json.load(f)
        else:
            records[label] = {}

    metrics_keys = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
        ("mAP@0.50", "mAP50"),
        ("mAP@0.50:0.95", "mAP50_95"),
    ]

    table_rows = []
    for metric_title, key in metrics_keys:
        row = {"Metric": metric_title}
        for exp_label in records.keys():
            val = records[exp_label].get(key)
            row[exp_label] = f"{val * 100:.2f}%" if val is not None else "Pending"
        table_rows.append(row)

    df = pd.DataFrame(table_rows)
    df.to_csv(comp_dir / "comparison_table.csv", index=False)
    
    with open(comp_dir / "comparison_table.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False) + "\n")

    print(f"\n[Performance Comparison Across Experiments]\n{df.to_markdown(index=False)}\n")

    available_exps = [e for e in records.keys() if records[e]]
    if len(available_exps) >= 2:
        m_titles = [m[0] for m in metrics_keys]
        x = np.arange(len(m_titles))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, exp_lbl in enumerate(available_exps):
            vals = [records[exp_lbl].get(m[1], 0.0) * 100 for m in metrics_keys]
            ax.bar(x + (i - len(available_exps) / 2 + 0.5) * width, vals, width, label=exp_lbl)

        ax.set_ylabel("Score (%)")
        ax.set_title("Detection Performance Across Experimental Pipelines", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(m_titles)
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_p = comp_dir / "comparison_metrics_chart.png"
        plt.savefig(chart_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Benchmark] Comparison chart saved -> {chart_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualitative", action="store_true", help="Generate qualitative predictions grid")
    parser.add_argument("--compare", action="store_true", help="Compile benchmark performance comparison")
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    if args.compare:
        compile_benchmark_comparison(Path(args.results_root))