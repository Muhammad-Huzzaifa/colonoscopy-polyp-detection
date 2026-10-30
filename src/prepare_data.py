import argparse
import json
import random
import shutil
import requests
import zipfile
from pathlib import Path
from typing import Dict, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.preprocessing import apply_preprocessing, save_preprocessing_comparison_grid

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download_kvasir_raw(data_root: Path) -> Path:
    """
    downloads and extracts the Kvasir-SEG dataset if not already present in the specified data_root directory

    Args:
        data_root (Path): the root directory where the Kvasir-SEG dataset will be downloaded and extracted

    Returns:
        Path: the path to the extracted Kvasir-SEG dataset directory
    """
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = raw_dir / "Kvasir-SEG"

    if (extracted_dir / "kavsir_bboxes.json").exists():
        return extracted_dir

    zip_path = raw_dir / "kvasir-seg.zip"
    if not zip_path.exists():
        print(f"[Data] Downloading Kvasir-SEG (~44 MB) from {KVASIR_URL} ...")
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(KVASIR_URL, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            print("[Data] SSL certificate verification failed on local CA. Retrying with unverified context...")
            response = requests.get(KVASIR_URL, headers=headers, stream=True, verify=False, timeout=60)
            response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            desc="[Data] kvasir-seg.zip",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print("[Data] Extracting Kvasir-SEG archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink(missing_ok=True)

    if not (extracted_dir / "kavsir_bboxes.json").exists():
        matches = list(raw_dir.rglob("kavsir_bboxes.json"))
        if matches:
            extracted_dir = matches[0].parent
    return extracted_dir


def build_base_yolo_dataset(
    data_root: Path,
    results_root: Path,
    seed: int = 42,
    splits: tuple = (0.8, 0.1, 0.1),
) -> Path:
    """
    creates the baseline YOLO dataset (`<data_root>/kvasir_none`) with deterministic 80/10/10 train/val/test splits and exports dataset stats to `results/dataset_stats/`

    Args:
        data_root (Path): the root directory where the Kvasir-SEG dataset will be downloaded and extracted
        results_root (Path): the root directory where the dataset statistics and visualizations will be saved
        seed (int, optional): the random seed for reproducibility. Defaults to 42.
        splits (tuple, optional): the train/val/test split ratios. Defaults to (0.8, 0.1, 0.1).

    Returns:
        Path: the path to the base YOLO dataset directory (`<data_root>/kvasir_none`)
    """
    base_dir = data_root / "kvasir_none"
    yaml_path = base_dir / "dataset.yaml"

    if yaml_path.exists():
        _write_dataset_yaml(base_dir)
        return base_dir

    kvasir_root = download_kvasir_raw(data_root)
    bbox_file = kvasir_root / "kavsir_bboxes.json"
    with open(bbox_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    img_ids = sorted(list(annotations.keys()))
    rng = random.Random(seed)
    rng.shuffle(img_ids)

    n_total = len(img_ids)
    n_train = int(n_total * splits[0])
    n_val = int(n_total * splits[1])

    split_ids = {
        "train": img_ids[:n_train],
        "val": img_ids[n_train : n_train + n_val],
        "test": img_ids[n_train + n_val :],
    }

    box_counts_per_image = []
    norm_widths, norm_heights = [], []
    resolutions = set()
    total_boxes = 0

    for split_name, ids in split_ids.items():
        img_out = base_dir / "images" / split_name
        lbl_out = base_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in tqdm(ids, desc=f"[Data] Building baseline split '{split_name}'"):
            meta = annotations[img_id]
            h, w = float(meta["height"]), float(meta["width"])
            resolutions.add(f"{int(w)}x{int(h)}")

            src_img = kvasir_root / "images" / f"{img_id}.jpg"
            dst_img = img_out / f"{img_id}.jpg"
            shutil.copy2(src_img, dst_img)

            boxes = meta.get("bbox", [])
            box_counts_per_image.append(len(boxes))
            yolo_lines = []

            for box in boxes:
                xmin, ymin = float(box["xmin"]), float(box["ymin"])
                xmax, ymax = float(box["xmax"]), float(box["ymax"])

                xc = np.clip(((xmin + xmax) / 2.0) / w, 0.0, 1.0)
                yc = np.clip(((ymin + ymax) / 2.0) / h, 0.0, 1.0)
                bw = np.clip((xmax - xmin) / w, 0.0, 1.0)
                bh = np.clip((ymax - ymin) / h, 0.0, 1.0)

                norm_widths.append(bw)
                norm_heights.append(bh)
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                total_boxes += 1

            with open(lbl_out / f"{img_id}.txt", "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_lines) + "\n")

    _write_dataset_yaml(base_dir)
    _export_dataset_statistics(
        results_root=results_root,
        base_dir=base_dir,
        n_total=n_total,
        total_boxes=total_boxes,
        split_ids=split_ids,
        resolutions=resolutions,
        norm_widths=norm_widths,
        norm_heights=norm_heights,
        box_counts=box_counts_per_image,
        seed=seed,
    )
    return base_dir


def ensure_preprocessed_dataset(
    data_root: Path,
    results_root: Path,
    preproc_name: str,
    preproc_params: Dict[str, Any] | None = None,
    seed: int = 42,
) -> Path:
    """
    ensures the dataset for `preproc_name` exists at `<data_root>/kvasir_<preproc_name>` and applies the specified preprocessing technique to the images, also writes the dataset.yaml file for YOLO training

    Args:
        data_root (Path): the root directory where the Kvasir-SEG dataset will be downloaded and extracted
        results_root (Path): the root directory where the dataset statistics and visualizations will be saved
        preproc_name (str): the name of the preprocessing technique to apply
        preproc_params (Dict[str, Any] | None, optional): the parameters for the preprocessing technique. Defaults to None.
        seed (int, optional): the random seed for reproducibility. Defaults to 42.

    Returns:
        Path: the path to the generated dataset.yaml file
    """
    base_dir = build_base_yolo_dataset(data_root, results_root, seed=seed)
    if preproc_name == "none":
        _write_dataset_yaml(base_dir)
        return base_dir / "dataset.yaml"

    target_dir = data_root / f"kvasir_{preproc_name}"
    target_yaml = target_dir / "dataset.yaml"

    if target_yaml.exists():
        _write_dataset_yaml(target_dir)
        return target_yaml

    print(f"[Data] Generating preprocessed dataset view: kvasir_{preproc_name} ...")
    for split_name in ("train", "val", "test"):
        src_img_dir = base_dir / "images" / split_name
        dst_img_dir = target_dir / "images" / split_name
        src_lbl_dir = base_dir / "labels" / split_name
        dst_lbl_dir = target_dir / "labels" / split_name

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        if dst_lbl_dir.exists():
            shutil.rmtree(dst_lbl_dir)
        shutil.copytree(src_lbl_dir, dst_lbl_dir)

        img_paths = sorted(list(src_img_dir.glob("*.jpg")))
        for img_path in tqdm(img_paths, desc=f"[Preproc:{preproc_name}] {split_name}"):
            bgr = cv2.imread(str(img_path))
            proc_bgr = apply_preprocessing(bgr, preproc_name, preproc_params)
            cv2.imwrite(
                str(dst_img_dir / img_path.name),
                proc_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )

    _write_dataset_yaml(target_dir)
    return target_yaml


def _write_dataset_yaml(dataset_dir: Path) -> None:
    """
    writes the dataset.yaml file for the specified dataset directory

    Args:
        dataset_dir (Path): the directory where the dataset.yaml file will be created
    """
    yaml_text = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['polyp']\n"
    )
    with open(dataset_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


def _export_dataset_statistics(
    results_root: Path,
    base_dir: Path,
    n_total: int,
    total_boxes: int,
    split_ids: Dict[str, list],
    resolutions: set,
    norm_widths: list,
    norm_heights: list,
    box_counts: list,
    seed: int,
) -> None:
    """
    generates and saves dataset statistics and visualizations to the specified results_root directory

    Args:
        results_root (Path): the root directory where the dataset statistics and visualizations will be saved
        base_dir (Path): the base directory of the dataset for which statistics are being generated
        n_total (int): the total number of images in the dataset
        total_boxes (int): the total number of annotated polyps in the dataset
        split_ids (Dict[str, list]): a dictionary containing the train/val/test split IDs
        resolutions (set): the set of unique image resolutions in the dataset
        norm_widths (list): the list of normalized widths of the bounding boxes
        norm_heights (list): the list of normalized heights of the bounding boxes
        box_counts (list): the list of the number of bounding boxes in each image
        seed (int): the random seed used for splitting the dataset
    """
    stats_dir = results_root / "dataset_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": "Kvasir-SEG",
        "source": "SimulaMet (https://datasets.simula.no/kvasir-seg/)",
        "total_images": n_total,
        "total_annotated_polyps": total_boxes,
        "multi_polyp_images": int(sum(1 for c in box_counts if c > 1)),
        "splits": {k: len(v) for k, v in split_ids.items()},
        "unique_resolutions_count": len(resolutions),
        "class_mapping": {"0": "polyp"},
        "seed": seed,
    }
    with open(stats_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(norm_widths, norm_heights, alpha=0.35, s=16, color="#0072B2")
    axes[0].set_title("Normalized Polyp Bounding Box Dimensions", fontweight="bold")
    axes[0].set_xlabel("Normalized Width (w / W)")
    axes[0].set_ylabel("Normalized Height (h / H)")
    axes[0].grid(alpha=0.3)

    areas = np.array(norm_widths) * np.array(norm_heights)
    axes[1].hist(areas, bins=30, color="#009E73", edgecolor="black", alpha=0.8)
    axes[1].set_title("Polyp Relative Area Distribution (w * h)", fontweight="bold")
    axes[1].set_xlabel("Fraction of Total Image Area")
    axes[1].set_ylabel("Number of Polyp Instances")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(stats_dir / "bbox_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    sample_imgs = sorted(list((base_dir / "images" / "test").glob("*.jpg")))[:3]
    if sample_imgs:
        save_preprocessing_comparison_grid(
            sample_imgs, stats_dir / "preprocessing_comparison_samples.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Kvasir-SEG YOLO datasets")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build raw baseline plus both clahe_lab and unsharp_laplacian datasets",
    )
    args = parser.parse_args()

    d_root = Path(args.data_root)
    r_root = Path(args.results_root)

    ensure_preprocessed_dataset(d_root, r_root, "none")
    if args.all:
        ensure_preprocessed_dataset(
            d_root, r_root, "clahe_lab", {"clip_limit": 2.0, "tile_grid_size": 8}
        )
        ensure_preprocessed_dataset(
            d_root, r_root, "unsharp_laplacian", {"strength": 1.2, "laplacian_weight": 0.15}
        )