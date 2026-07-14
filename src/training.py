"""
Entry point de chay training.

Vi du:
    python train_main.py --data_dir ./data --epochs 100 --batch_size 16 --nc 80

Sau khi pretrain xong, dung checkpoints/best_trunk.pt de nap lai backbone+neck
cho model voi HEAD moi (xem huong dan trong README phan "Doi HEAD sau pretrain").
"""
import os
import sys

# Them thu muc goc vao sys.path de ho tro import ca dang 'from src.xxx import ...' va import truc tiep
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
from dataclasses import fields
from src.config import TrainConfig
from src.train.engine import run_training

def parse_args():
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        # Kiem tra xem field co kieu tuple/list khong
        is_tuple = False
        if isinstance(f.default, (tuple, list)):
            is_tuple = True
        elif hasattr(f.type, "__origin__") and f.type.__origin__ in (tuple, list):
            is_tuple = True
        elif isinstance(f.type, str) and ("Tuple" in f.type or "tuple" in f.type or "List" in f.type or "list" in f.type):
            is_tuple = True

        if f.type in (int, float, str):
            parser.add_argument(f"--{f.name}", type=f.type, default=None)
        elif f.type == bool:
            parser.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
        elif is_tuple:
            parser.add_argument(f"--{f.name}", type=str, default=None, help=f"Tuple values (e.g. '1,2,3' or '1 2 3')")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig()
    for f in fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            # Kiem tra xem field co kieu tuple/list khong de parse cho dung dinh dang
            is_tuple = False
            if isinstance(f.default, (tuple, list)):
                is_tuple = True
            elif hasattr(f.type, "__origin__") and f.type.__origin__ in (tuple, list):
                is_tuple = True
            elif isinstance(f.type, str) and ("Tuple" in f.type or "tuple" in f.type or "List" in f.type or "list" in f.type):
                is_tuple = True

            if is_tuple:
                element_type = int
                if isinstance(f.default, (tuple, list)) and len(f.default) > 0:
                    element_type = type(f.default[0])
                elif hasattr(f.type, "__args__") and f.type.__args__:
                    t = f.type.__args__[0]
                    if isinstance(t, type):
                        element_type = t
                
                parts = v.replace(",", " ").split()
                parsed_val = tuple(element_type(x) for x in parts)
                setattr(cfg, f.name, parsed_val)
            else:
                setattr(cfg, f.name, v)

    print("==== Training config ====")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print("==========================")

    run_training(cfg)

if __name__ == "__main__":
    main()