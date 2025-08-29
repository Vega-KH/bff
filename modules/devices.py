import sys
import torch

def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    # A placeholder for mac_specific.has_mps
    return torch.backends.mps.is_available()

def get_optimal_device_name():
    if torch.cuda.is_available():
        return "cuda"

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())

cpu: torch.device = torch.device("cpu")
device: torch.device = get_optimal_device()
dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
