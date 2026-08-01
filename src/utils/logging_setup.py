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


class FlushFileHandler(logging.FileHandler):
    """Custom FileHandler tu dong flush du lieu xuong đia ngay sau moi dong log."""
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(
    log_dir: str = "./logs",
    run_name: str = "finetune",
    level: int = logging.INFO,
    also_stdout: bool = False,
) -> logging.Logger:
    """
    Tao logger ghi ra file `{log_dir}/{run_name}_{timestamp}.log`.
    
    Gan Handler vao Root Logger de TAT CA cac module trong project 
    (nhu logging.getLogger('finetune'), logging.getLogger('train'), ...)
    deu tu dong ghi chung vao 1 file log.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{run_name}_{timestamp}.log")

    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Lay Root Logger de tat ca cac file/module deu ghi log vao day
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Xoa cac handlers cu neu da ton tai (tranh ghi trung lach log khi goi lai)
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Dung FlushFileHandler de ép ghi realtime xuong o đia
    fh = FlushFileHandler(log_filename, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root_logger.addHandler(sh)

    # Lay logger theo run_name de in dong thong bao khoi tao
    logger = logging.getLogger(run_name)
    logger.info("=" * 60)
    logger.info("Logger khoi tao thanh cong.")
    logger.info(f"File log : {log_filename}")
    logger.info("=" * 60)

    return logger