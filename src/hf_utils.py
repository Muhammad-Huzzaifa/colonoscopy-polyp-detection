# src/hf_utils.py
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()


def push_model_to_hf(
    weights_path: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    exp_dir: Path,
) -> Optional[str]:
    """
    pushes the trained model weights and a generated Model Card to Hugging Face Hub

    Args:
        weights_path (Path): path to the trained model weights file (e.g., best.pt)
        config (Dict[str, Any]): configuration dictionary containing experiment and model details
        metrics (Dict[str, Any]): dictionary containing evaluation metrics for the model
        exp_dir (Path): path to the local experiment results directory

    Returns:
        Optional[str]: the URL of the uploaded model on Hugging Face Hub, or None if the upload fails
    """
    token = os.getenv("HF_TOKEN")
    repo_id = config.get("huggingface", {}).get("repo_id") or os.getenv("HF_REPO_ID")

    if not token or not repo_id:
        print("[HF Hub] Warning: HF_TOKEN or HF_REPO_ID not set. Skipping Hugging Face upload.")
        return None

    if not weights_path.exists():
        print(f"[HF Hub] Warning: Checkpoint not found at {weights_path}. Skipping upload.")
        return None

    exp_id = config.get("experiment_id", "baseline")
    api = HfApi(token=token)

    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"[HF Hub] Connected to repo: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[HF Hub] Error creating/accessing repo: {e}")
        return None

    dest_path_in_repo = f"{exp_id}/best.pt"
    print(f"[HF Hub] Uploading {weights_path.name} to {repo_id}/{dest_path_in_repo} ...")

    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo=dest_path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )

    card_content = _generate_model_card(config, metrics, dest_path_in_repo)
    card_path_in_repo = f"{exp_id}/README.md"
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo=card_path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )

    hub_url = f"https://huggingface.co/{repo_id}/blob/main/{dest_path_in_repo}"
    
    link_info = {
        "experiment_id": exp_id,
        "repo_id": repo_id,
        "hub_file": dest_path_in_repo,
        "download_url": hub_url,
    }
    with open(exp_dir / "hf_checkpoint_link.json", "w", encoding="utf-8") as f:
        json.dump(link_info, f, indent=2)

    print(f"[HF Hub] Upload complete! Model available at: {hub_url}")
    return hub_url


def _generate_model_card(
    config: Dict[str, Any], metrics: Dict[str, Any], path_in_repo: str
) -> str:
    """
    generates a Model Card for the uploaded model

    Args:
        config (Dict[str, Any]): configuration dictionary containing experiment and model details
        metrics (Dict[str, Any]): dictionary containing evaluation metrics for the model
        path_in_repo (str): the path to the model weights file in the Hugging Face repository

    Returns:
        str: the content of the Model Card in Markdown format
    """
    exp_name = config.get("experiment_name", "Polyp Detection")
    model_arch = config.get("model", {}).get("architecture", "yolov8m")
    preproc = config.get("preprocessing", {}).get("description", "None")

    lines = [
        "---",
        "tags:",
        "- medical-imaging",
        "- object-detection",
        "- polyp-detection",
        "- colonoscopy",
        "- ultralytics",
        "datasets:",
        "- kvasir-seg",
        "---",
        f"# {exp_name}",
        "",
        "## Model Details",
        f"- **Task:** Polyp Detection in Colonoscopy Images (Single-Class)",
        f"- **Architecture:** `{model_arch}`",
        f"- **Preprocessing Applied:** {preproc}",
        f"- **Weights File:** `{path_in_repo}`",
        "",
        "## Test Set Evaluation Metrics",
        f"- **Precision:** {metrics.get('precision', 0.0) * 100:.2f}%",
        f"- **Recall:** {metrics.get('recall', 0.0) * 100:.2f}%",
        f"- **F1-Score:** {metrics.get('f1_score', 0.0) * 100:.2f}%",
        f"- **mAP@50:** {metrics.get('mAP50', 0.0) * 100:.2f}%",
        f"- **mAP@50-95:** {metrics.get('mAP50_95', 0.0) * 100:.2f}%",
        "",
        "## Hyperparameters",
        "```yaml",
        json.dumps(config.get("training", {}), indent=2),
        "```",
    ]
    return "\n".join(lines)