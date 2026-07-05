"""
Entry point de chay training.

Vi du:
    python train_main.py --data_dir ./data --epochs 100 --batch_size 16 --nc 80

Sau khi pretrain xong, dung checkpoints/best_trunk.pt de nap lai backbone+neck
cho model voi HEAD moi (xem huong dan trong README phan "Doi HEAD sau pretrain").
"""
import argparse
from dataclasses import fields
from config import TrainConfig
from train.engine import run_training

def parse_args():
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        default = f.default
        if f.type in (int, float, str):
            parser.add_argument(f"--{f.name}", type=f.type, default=None)
        elif f.type == bool:
            parser.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
        # cac truong dang Tuple (backbone_w, backbone_n, strides) giu nguyen default,
        # neu muon chinh thi sua truc tiep trong config.py hoac truyen TrainConfig() tuy chinh.
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    print("==== Training config ====")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print("==========================")

    run_training(cfg)

if __name__ == "__main__":
    main()