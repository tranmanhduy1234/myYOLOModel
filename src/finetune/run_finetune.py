import logging

from src.finetune.finetune_config import FinetuneConfig
from src.finetune.finetune_engine import run_finetune
from src.utils.logging_setup import setup_logging

def main():
    cfg = FinetuneConfig()

    setup_logging(
        log_dir=cfg.log_dir,
        run_name=cfg.run_name,
        also_stdout=False,
    )
    logger = logging.getLogger("finetune")

    logger.info("==== Finetune Config ====")
    for k, v in cfg.__dict__.items():
        logger.info(f"  {k}: {v}")
    logger.info("=========================")

    run_finetune(cfg)

if __name__ == "__main__":
    main()