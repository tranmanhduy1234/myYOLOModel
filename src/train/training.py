import logging

from src.config import TrainConfig
from src.train.engine import run_training
from src.utils.logging_setup import setup_logging


def main():
    cfg = TrainConfig()
    setup_logging(
        log_dir=cfg.log_dir,
        run_name=cfg.run_name,
        also_stdout=False,
    )
    logger = logging.getLogger("train")

    logger.info("==== Training config ====")
    for k, v in cfg.__dict__.items():
        logger.info(f"  {k}: {v}")
    logger.info("==========================")
    run_training(cfg)

if __name__ == "__main__":
    main()