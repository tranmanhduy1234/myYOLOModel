"""
seed.py
=======
Tien ich dat seed de dam bao tinh tai lap (reproducibility) cho training.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Dat seed cho Python random, NumPy va PyTorch (CPU + tat ca GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)