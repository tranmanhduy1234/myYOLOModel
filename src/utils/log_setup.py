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
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{run_name}_{timestamp}.log")
    # ------- format -------
    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # ------- root logger -------
    logger = logging.getLogger("train")
    logger.setLevel(level)
    # Xoá handler cũ nếu logger đã tồn tại (tránh duplicate khi gọi lại)
    if logger.handlers:
        logger.handlers.clear()
    # ------- file handler (chính) -------
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # ------- stdout handler (tuỳ chọn) -------
    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Không propagate lên root để tránh log hiện ở console ngoài ý muốn
    logger.propagate = False

    logger.info("=" * 60)
    logger.info("Logger khởi tạo thành công.")
    logger.info(f"File log : {log_filename}")
    logger.info("=" * 60)

    return logger