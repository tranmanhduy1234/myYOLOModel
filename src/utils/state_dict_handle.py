import random
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, best_val, cfg):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "cfg": cfg.__dict__
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, schedular=None, ema=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if schedular is not None and "scheduler" in ckpt:
        schedular.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))

def load_model_only(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])