# train.py
import argparse
from pathlib import Path
from typing import Any, Dict
import yaml

from src.evaluate import evaluate_and_export_experiment
from src.hf_utils import push_model_to_hf
from src.models import get_model
from src.prepare_data import ensure_preprocessed_dataset


def run_experiment(
    config_path: Path,
    data_root: Path = Path("data"),
    results_root: Path = Path("results"),
    push_to_hf: bool = True,
) -> Dict[str, Any]:
    """
    runs a complete experiment pipeline including data preprocessing, model training, evaluation, and optional upload to Hugging Face Hub
    
    Args:
        config_path (Path): path to the YAML configuration file specifying experiment parameters
        data_root (Path, optional): path to the root directory containing the raw dataset. Defaults to Path("data").
        results_root (Path, optional): path to the root directory where the experiment results will be stored. Defaults to Path("results").
        push_to_hf (bool, optional): whether to push the trained model and artifacts to the Hugging Face Hub. Defaults to True.

    Returns:
        Dict[str, Any]: _description_
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_id = config.get("experiment_id", "baseline")
    print(f" Starting Experiment: {exp_id}")
    print(f" Description: {config.get('experiment_name')}")

    p_cfg = config.get("preprocessing", {})
    p_name = p_cfg.get("name", "none")
    p_params = p_cfg.get("params", {})
    t_cfg = config.get("training", {})

    dataset_yaml = ensure_preprocessed_dataset(
        data_root=data_root,
        results_root=results_root,
        preproc_name=p_name,
        preproc_params=p_params,
        seed=t_cfg.get("seed", 42),
    )

    model = get_model(config)

    run_project = Path("runs")
    run_name = exp_id

    train_args = {
        "data": str(dataset_yaml.resolve()),
        "epochs": t_cfg.get("epochs", 80),
        "batch": t_cfg.get("batch_size", 16),
        "imgsz": config.get("model", {}).get("imgsz", 640),
        "optimizer": t_cfg.get("optimizer", "AdamW"),
        "lr0": t_cfg.get("lr0", 0.001),
        "lrf": t_cfg.get("lrf", 0.01),
        "weight_decay": t_cfg.get("weight_decay", 0.0005),
        "cos_lr": t_cfg.get("cos_lr", True),
        "patience": t_cfg.get("patience", 20),
        "seed": t_cfg.get("seed", 42),
        "workers": t_cfg.get("workers", 4),
        "project": str(run_project),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    print(f"[Train] Launching YOLOv8 training ({train_args['epochs']} epochs) ...")
    model.train(**train_args)

    if hasattr(model, "trainer") and model.trainer is not None:
        run_dir = Path(model.trainer.save_dir)
        best_weights = Path(model.trainer.best) if hasattr(model.trainer, "best") else run_dir / "weights" / "best.pt"
    else:
        run_dir = (run_project / "detect" / run_name).resolve()
        best_weights = run_dir / "weights" / "best.pt"

    eval_model = get_model(config)
    eval_model = eval_model.__class__(best_weights)

    metrics = evaluate_and_export_experiment(
        model=eval_model,
        dataset_yaml=dataset_yaml,
        config=config,
        run_dir=run_dir,
        results_root=results_root,
    )

    if push_to_hf and config.get("huggingface", {}).get("push_weights", True):
        push_model_to_hf(
            weights_path=best_weights,
            config=config,
            metrics=metrics,
            exp_dir=results_root / exp_id,
        )

    print(f"\n[Completed] Experiment {exp_id} successfully finished!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local polyp detection experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--no-hf", action="store_true", help="Skip pushing to Hugging Face Hub")
    args = parser.parse_args()

    run_experiment(
        config_path=Path(args.config),
        data_root=Path(args.data_root),
        results_root=Path(args.results_root),
        push_to_hf=not args.no_hf,
    )