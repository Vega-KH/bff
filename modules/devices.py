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
device_gfpgan: torch.device = get_optimal_device()
device_codeformer: torch.device = get_optimal_device()
device_esrgan: torch.device = get_optimal_device()

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class without_autocast(object):
    def __init__(self, device_type='cuda', dtype=torch.float16, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        if self.enabled:
            torch.set_autocast_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            torch.set_autocast_enabled(self.prev)
        return False
