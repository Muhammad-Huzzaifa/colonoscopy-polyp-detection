# src/models.py
from pathlib import Path
from typing import Any, Dict
from ultralytics import YOLO


def get_model(config: Dict[str, Any]) -> Any:
    """
    factory function to initialize detection models

    Args:
        config (Dict[str, Any]): configuration dictionary containing model parameters such as architecture and type

    Raises:
        ValueError: if the specified model type is not supported

    Returns:
        Any: an instance of the initialized model (e.g., YOLO or RT-DETR)
    """
    model_cfg = config.get("model", {})
    arch = model_cfg.get("architecture", "yolov8m.pt")
    model_type = model_cfg.get("type", "yolo").lower()

    if model_type == "yolo":
        model = YOLO(arch)
        return model

    elif model_type in ("rtdetr", "transformer"):
        model = YOLO(arch)
        return model

    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose 'yolo' or 'rtdetr'.")