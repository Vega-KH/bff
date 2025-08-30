from __future__ import annotations
import cv2
import numpy as np
from typing import overload, Literal
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

def load_image_as_float_bgr(path: str) -> np.ndarray:
    """
    Load an image from disk as float32 BGR in range [0,1].
    """
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image_bgr.astype(np.float32) / 255.0

def bgr_to_hsv_unit(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert float32 BGR ([0,1]) to HSV with all channels normalized to [0,1].
    OpenCV returns H in [0,360] for float32, S,V in [0,1]. We normalize H to [0,1].
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Normalize hue from [0,360] to [0,1]
    hsv[..., 0] = hsv[..., 0] / 360.0
    return hsv

def _validate_same_shape(*images: np.ndarray) -> None:
    """
    Ensure all provided images have identical shape.
    """
    if len(images) < 2:
        return
    base_shape = images[0].shape
    for i, img in enumerate(images[1:], start=1):
        if img.shape != base_shape:
            raise ValueError(
                f"All images must have the same shape. Found mismatch: "
                f"{base_shape} vs {img.shape} (index {i})"
            )

def compute_hsv_mse(
    original_hsv: np.ndarray,
    test_hsv: np.ndarray,
) -> np.ndarray:
    """
    Compute Mean Squared Error (MSE) in HSV space per pixel.
    Hue wrap-around is handled in normalized [0,1] hue space.
    Returns a 2D float32 array of MSE scores.
    """
    # Absolute channel diffs
    hue_diff = np.abs(original_hsv[:, :, 0] - test_hsv[:, :, 0])
    hue_diff = np.minimum(hue_diff, 1.0 - hue_diff)  # wrap-around on unit circle
    sat_diff = np.abs(original_hsv[:, :, 1] - test_hsv[:, :, 1])
    val_diff = np.abs(original_hsv[:, :, 2] - test_hsv[:, :, 2])

    # MSE calculation
    mse = (np.square(hue_diff) + np.square(sat_diff) + np.square(val_diff)) / 3.0
    return mse.astype(np.float32)


def compute_ssim_error_map(
    original_bgr: np.ndarray,
    test_bgr: np.ndarray,
) -> np.ndarray:
    """
    Computes a structural dissimilarity map using (1 - SSIM).
    Returns a 2D float32 array of structural error scores.
    """
    # Convert to grayscale for SSIM
    original_gray = rgb2gray(original_bgr)
    test_gray = rgb2gray(test_bgr)
    
    # Compute full SSIM map. data_range is 1.0 as images are in [0,1]
    # We use a smaller win_size for more localized comparison on 512x512 faces
    _, ssim_map = ssim(original_gray, test_gray, full=True, data_range=1.0, win_size=11)  # type: ignore
    
    # Return structural error map
    return (1.0 - ssim_map).astype(np.float32)


@dataclass
class ErrorMaps:
    """Container for pre-computed error maps."""
    mse_g: np.ndarray
    mse_c: np.ndarray
    ssim_err_g: np.ndarray
    ssim_err_c: np.ndarray


def precompute_error_maps(
    original_bgr: np.ndarray,
    gfpgan_bgr: np.ndarray,
    codeformer_bgr: np.ndarray,
) -> ErrorMaps:
    """
    Performs the expensive MSE and SSIM calculations.
    """
    # --- Color Error (MSE) ---
    original_hsv = bgr_to_hsv_unit(original_bgr)
    gfpgan_hsv = bgr_to_hsv_unit(gfpgan_bgr)
    codeformer_hsv = bgr_to_hsv_unit(codeformer_bgr)
    mse_g = compute_hsv_mse(original_hsv, gfpgan_hsv)
    mse_c = compute_hsv_mse(original_hsv, codeformer_hsv)

    # --- Structural Error (1 - SSIM) ---
    ssim_err_g = compute_ssim_error_map(original_bgr, gfpgan_bgr)
    ssim_err_c = compute_ssim_error_map(original_bgr, codeformer_bgr)
    
    return ErrorMaps(mse_g=mse_g, mse_c=mse_c, ssim_err_g=ssim_err_g, ssim_err_c=ssim_err_c)


@overload
def merge_images_hybrid(
    original_bgr: np.ndarray,
    gfpgan_bgr: np.ndarray,
    codeformer_bgr: np.ndarray,
    error_maps: ErrorMaps,
    *,
    alpha: float = 0.5,
    delta_threshold: float = ...,
    tie_epsilon: float = ...,
    return_stats: Literal[False] = False,
) -> np.ndarray: ...

@overload
def merge_images_hybrid(
    original_bgr: np.ndarray,
    gfpgan_bgr: np.ndarray,
    codeformer_bgr: np.ndarray,
    error_maps: ErrorMaps,
    *,
    alpha: float = 0.5,
    delta_threshold: float = ...,
    tie_epsilon: float = ...,
    return_stats: Literal[True],
) -> tuple[np.ndarray, dict[str, int]]: ...

def merge_images_hybrid(
    original_bgr: np.ndarray,
    gfpgan_bgr: np.ndarray,
    codeformer_bgr: np.ndarray,
    error_maps: ErrorMaps,
    *,
    alpha: float = 0.5,
    delta_threshold: float = 0.1,
    tie_epsilon: float = 1e-3,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """
    Merge two enhanced images using a hybrid score of pre-computed error maps.
    - All images must be float32 BGR in [0,1] and same shape.
    - `alpha` blends between color (1.0) and structure (0.0).
    - If both deltas > delta_threshold: fallback to original pixel.
    - If |delta_g - delta_c| <= tie_epsilon: blend average for smoother seams.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")
    
    # --- Exact Match Optimization ---
    atol = 1e-6
    exact_g = np.isclose(original_bgr, gfpgan_bgr, rtol=0.0, atol=atol).all(axis=2)
    exact_c = np.isclose(original_bgr, codeformer_bgr, rtol=0.0, atol=atol).all(axis=2)
    exact_match_mask = exact_g & exact_c

    # --- Hybrid Score ---
    delta_g = (alpha * error_maps.mse_g) + ((1 - alpha) * error_maps.ssim_err_g)
    delta_c = (alpha * error_maps.mse_c) + ((1 - alpha) * error_maps.ssim_err_c)

    # --- Masking and Selection ---
    g_is_better = delta_g < delta_c
    tie_mask = np.abs(delta_g - delta_c) <= float(tie_epsilon)
    both_bad_mask = (delta_g > float(delta_threshold)) & (delta_c > float(delta_threshold))

    # Derived masks for stats (exclude exact matches; threshold-kept excludes exact too)
    g_non_tie_mask = g_is_better & ~tie_mask
    c_non_tie_mask = (~g_is_better) & ~tie_mask
    gf_used_mask = g_non_tie_mask & ~both_bad_mask & ~exact_match_mask
    cf_used_mask = c_non_tie_mask & ~both_bad_mask & ~exact_match_mask
    kept_original_threshold_mask = both_bad_mask & ~exact_match_mask

    stats: dict[str, int] | None = None
    if return_stats:
        stats = {
            "identical_kept": int(np.count_nonzero(exact_match_mask)),
            "from_gfpgan": int(np.count_nonzero(gf_used_mask)),
            "from_codeformer": int(np.count_nonzero(cf_used_mask)),
            "kept_original_threshold": int(np.count_nonzero(kept_original_threshold_mask)),
            "ties_blended": int(np.count_nonzero(tie_mask & ~exact_match_mask)),
        }

    # Initialize output
    final_bgr = np.empty_like(original_bgr)

    # Non-tie selections
    final_bgr[g_non_tie_mask] = gfpgan_bgr[g_non_tie_mask]
    final_bgr[c_non_tie_mask] = codeformer_bgr[c_non_tie_mask]

    # Ties: average
    if np.any(tie_mask):
        avg = 0.5 * (gfpgan_bgr + codeformer_bgr)
        final_bgr[tie_mask] = avg[tie_mask]

    # Fallback to original if both are too far from original
    if np.any(both_bad_mask):
        final_bgr[both_bad_mask] = original_bgr[both_bad_mask]

    # Exact-match optimization last
    if np.any(exact_match_mask):
        final_bgr[exact_match_mask] = original_bgr[exact_match_mask]

    if return_stats:
        assert stats is not None
        return final_bgr, stats
    return final_bgr