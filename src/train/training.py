from src.config import TrainConfig
from src.train.engine import run_training

def main():
    cfg = TrainConfig()

    print("==== Training config ====")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print("==========================")
    run_training(cfg)

if __name__ == "__main__":
    main()