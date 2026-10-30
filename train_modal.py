# train_modal.py
import os
from pathlib import Path
import modal

app = modal.App("colonoscopy-polyp-detection")
volume = modal.Volume.from_name("polyp-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "curl")
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.9.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0.1",
        "huggingface-hub>=0.24.0",
        "python-dotenv>=1.0.1",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "tabulate>=0.9.0",
    )
    .add_local_python_source("src")
    .add_local_file("train.py", "/root/train.py")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def train_remote(config_dict: dict) -> dict:
    """
    train a model remotely on a cloud GPU

    Args:
        config_dict (dict): a dictionary containing the experiment configuration

    Returns:
        dict: a dictionary containing the experiment metrics and result files
    """
    import yaml
    from train import run_experiment

    vol_data_root = Path("/vol/data")
    vol_data_root.mkdir(parents=True, exist_ok=True)
    local_results = Path("/root/results")

    tmp_cfg_path = Path("/root/active_config.yaml")
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    metrics = run_experiment(
        config_path=tmp_cfg_path,
        data_root=vol_data_root,
        results_root=local_results,
        push_to_hf=True,
    )

    volume.commit()

    exp_id = config_dict.get("experiment_id", "exp")
    exp_dir = local_results / exp_id
    exported_files = {}

    for file_path in exp_dir.glob("*"):
        if file_path.is_file():
            exported_files[file_path.name] = file_path.read_bytes()

    return {"metrics": metrics, "files": exported_files}


@app.local_entrypoint()
def main(config: str = "configs/exp1_baseline.yaml"):
    """
    main entry point for the local training script

    Args:
        config (str, optional): path to the configuration file. Defaults to "configs/exp1_baseline.yaml".

    Raises:
        FileNotFoundError: if the specified configuration file does not exist
    """
    import yaml

    cfg_path = Path(config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {config} not found.")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    exp_id = config_dict.get("experiment_id", "exp")
    print(f"[Modal] Submitting {exp_id} to cloud GPU ...")

    result = train_remote.remote(config_dict)

    local_exp_dir = Path("results") / exp_id
    local_exp_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in result["files"].items():
        (local_exp_dir / filename).write_bytes(content)

    print(f"\n[Modal] Successfully pulled artifacts to: {local_exp_dir}")
    print(f"[Modal] Metrics Summary: {result['metrics']}")