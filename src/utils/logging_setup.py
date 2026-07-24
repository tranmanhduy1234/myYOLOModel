"""
logging_setup.py
=================
Cau hinh Python `logging` (ghi file .log + tuy chon in ra stdout).

Luu y: day la text/console logging, khac voi TensorBoard logging (xem tb_logger.py).
"""

import logging
import os
import sys
from datetime import datetime


def setup_logging(
    log_dir: str = "./logs",
    run_name: str = "train",
    level: int = logging.INFO,
    also_stdout: bool = False,
) -> logging.Logger:
    """Tao logger "train" ghi ra file `{log_dir}/{run_name}_{timestamp}.log`."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{run_name}_{timestamp}.log")

    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("train")
    logger.setLevel(level)
    # Xoa handler cu neu logger da ton tai (tranh duplicate khi goi lai)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Khong propagate len root de tranh log hien o console ngoai y muon
    logger.propagate = False

    logger.info("=" * 60)
    logger.info("Logger khoi tao thanh cong.")
    logger.info(f"File log : {log_filename}")
    logger.info("=" * 60)

    return logger