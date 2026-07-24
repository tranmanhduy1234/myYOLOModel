from typing import Optional, Tuple

import torch

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema,
    epoch: int,
    best_val: float,
    cfg,
) -> None:
    """Luu toan bo trang thai training vao 1 file .pt."""
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "cfg": cfg.__dict__,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    ema=None,
    map_location: str = "cpu",
) -> Tuple[int, float]:
    """Nap checkpoint day du (model + optimizer + scheduler + ema neu co).

    Returns:
        (epoch, best_val) da luu trong checkpoint (mac dinh (0, inf) neu khong co).
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))

def load_model_only(path: str, model: torch.nn.Module, map_location: str = "cpu") -> None:
    """Chi nap trong so model (dung khi fine-tune / inference, khong can optimizer)."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])