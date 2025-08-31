import torch

def get_param(model) -> torch.Tensor | None:
    if isinstance(model, torch.nn.Module):
        for param in model.parameters():
            return param
    return None