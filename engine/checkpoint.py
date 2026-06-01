from pathlib import Path

import torch


def save_checkpoint(state: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)
