import cv2
import numpy as np
import torch


def bgr_image_to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert a BGR NumPy image in [0..1] range to a PyTorch RGB float32 tensor."""
    assert img.shape[2] == 3, "image must be RGB"
    if img.dtype == "float64":
        img = img.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def rgb_tensor_to_bgr_image(tensor: torch.Tensor, *, min_max=(0.0, 1.0)) -> np.ndarray:
    """
    Convert a PyTorch RGB tensor in range `min_max` to a BGR NumPy image in [0..1] range.
    """
    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    assert tensor.dim() == 3, "tensor must be RGB"
    img_np = tensor.numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image, no RGB/BGR required
        return np.squeeze(img_np, axis=2)
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)