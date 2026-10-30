# src/preprocessing.py
from pathlib import Path
from typing import Callable, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_none(image_bgr: np.ndarray, **kwargs) -> np.ndarray:
    """
    applies no preprocessing to the input image

    Args:
        image_bgr (np.ndarray): input image in BGR format

    Returns:
        np.ndarray: the same input image without any modifications
    """
    return image_bgr


def apply_clahe_lab(
    image_bgr: np.ndarray,
    gaussian_ksize: int = 3,
    gaussian_sigma: float = 0.5,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    **kwargs,
) -> np.ndarray:
    """
    1. applies a 3x3 Gaussian low-pass filter to attenuate high-frequency endoscopic sensor noise prior to local contrast amplification
    2. converts BGR -> CIE LAB to decouple Luminance (L) from Chrominance (a, b)
    3. applies Contrast Limited Adaptive Histogram Equalization (CLAHE) strictly on the L-channel to brighten dark mucosal folds without distorting tissue color

    Args:
        image_bgr (np.ndarray): input image in BGR format
        gaussian_ksize (int, optional): size of the Gaussian kernel. Defaults to 3.
        gaussian_sigma (float, optional): standard deviation of the Gaussian kernel. Defaults to 0.5.
        clip_limit (float, optional): clipping limit for CLAHE. Defaults to 2.0.
        tile_grid_size (int, optional): size of the tile grid for CLAHE. Defaults to 8.

    Returns:
        np.ndarray: the preprocessed image in BGR format after applying Gaussian smoothing and CLAHE on the L-channel
    """
    k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
    denoised = cv2.GaussianBlur(image_bgr, (k, k), sigmaX=gaussian_sigma)

    lab = cv2.cvtColor(denoised, cv2.cvtColor_BGR2LAB if hasattr(cv2, "cvtColor_BGR2LAB") else cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    l_enhanced = clahe.apply(l_chan)

    merged_lab = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_laplacian(
    image_bgr: np.ndarray,
    blur_ksize: int = 5,
    sigma: float = 1.2,
    strength: float = 1.2,
    laplacian_weight: float = 0.15,
    **kwargs,
) -> np.ndarray:
    """
    1. computes an unsharp mask via Gaussian smoothing: I_unsharp = I + strength * (I - G*I)
    2. computes second-order Laplacian edge response on the luminance channel to highlight subtle mucosal elevation boundaries of flat/sessile polyps

    Args:
        image_bgr (np.ndarray): input image in BGR format
        blur_ksize (int, optional): size of the Gaussian kernel for blurring. Defaults to 5.
        sigma (float, optional): standard deviation of the Gaussian kernel. Defaults to 1.2.
        strength (float, optional): strength of the unsharp mask. Defaults to 1.2.
        laplacian_weight (float, optional): weight of the Laplacian filter. Defaults to 0.15.

    Returns:
        np.ndarray: the preprocessed image in BGR format after applying unsharp masking and Laplacian enhancement
    """
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(image_bgr, (k, k), sigmaX=float(sigma))

    unsharp = cv2.addWeighted(
        image_bgr, 1.0 + float(strength), blurred, -float(strength), 0
    )

    if float(laplacian_weight) > 0:
        lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        lap = cv2.Laplacian(l_chan, cv2.CV_32F, ksize=3)
        l_boosted = np.clip(
            l_chan.astype(np.float32) - float(laplacian_weight) * lap, 0, 255
        ).astype(np.uint8)
        unsharp = cv2.cvtColor(cv2.merge((l_boosted, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    return unsharp


# registery of preprocessings
PREPROCESSORS: Dict[str, Callable[..., np.ndarray]] = {
    "none": apply_none,
    "clahe_lab": apply_clahe_lab,
    "unsharp_laplacian": apply_unsharp_laplacian,
}


def get_preprocessor(name: str) -> Callable[..., np.ndarray]:
    """
    retrieves a preprocessor function by its name

    Args:
        name (str): name of the preprocessor to retrieve

    Raises:
        ValueError: if the specified preprocessor name is not found in the registry

    Returns:
        Callable[..., np.ndarray]: the corresponding preprocessor function that takes an image and optional parameters and returns a preprocessed image
    """
    if name not in PREPROCESSORS:
        raise ValueError(
            f"Unknown preprocessor '{name}'. Available: {list(PREPROCESSORS.keys())}"
        )
    return PREPROCESSORS[name]


def apply_preprocessing(
    image_bgr: np.ndarray, name: str, params: Dict[str, Any] | None = None
) -> np.ndarray:
    """
    applies the specified preprocessing technique to the input image

    Args:
        image_bgr (np.ndarray): the input image in BGR format
        name (str): the name of the preprocessor to apply
        params (Dict[str, Any] | None, optional): the parameters for the preprocessor. Defaults to None.

    Returns:
        np.ndarray: the preprocessed image in BGR format
    """
    fn = get_preprocessor(name)
    return fn(image_bgr, **(params or {}))


def save_preprocessing_comparison_grid(
    sample_img_paths: list[Path], output_path: Path
) -> None:
    """
    generates a visual comparison grid of different preprocessing techniques applied to the sample images and saves it to the specified output path

    Args:
        sample_img_paths (list[Path]): list of paths to the sample images to be processed
        output_path (Path): the path where the comparison grid image will be saved
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(sample_img_paths)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    methods = [
        ("Exp 1: Raw Baseline", "none", {}),
        ("Exp 2: LAB-CLAHE + Gaussian", "clahe_lab", {"clip_limit": 2.0, "tile_grid_size": 8}),
        ("Exp 3: Unsharp + Laplacian", "unsharp_laplacian", {"strength": 1.2, "laplacian_weight": 0.15}),
    ]

    for row_idx, img_path in enumerate(sample_img_paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        hist_ax = axes[row_idx, 3]
        colors = ["#1f77b4", "#2ca02c", "#d62728"]

        for col_idx, (title, m_name, m_params) in enumerate(methods):
            proc_bgr = apply_preprocessing(bgr, m_name, m_params)
            proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

            ax = axes[row_idx, col_idx]
            ax.imshow(proc_rgb)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

            gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-6)
            hist_ax.plot(hist, color=colors[col_idx], label=title.split(":")[0], alpha=0.85)

        if row_idx == 0:
            hist_ax.set_title("Luminance Intensity Histogram", fontsize=11, fontweight="bold")
        hist_ax.set_xlim([0, 255])
        hist_ax.set_xlabel("Pixel Intensity [0-255]")
        hist_ax.set_ylabel("Normalized Frequency")
        hist_ax.legend(fontsize=8)
        hist_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Preprocessing] Saved visual comparison grid -> {output_path}")