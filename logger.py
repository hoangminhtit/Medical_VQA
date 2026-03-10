import logging
import os
from datetime import datetime


def setup_logger(log_dir: str = "logs", log_name: str = None) -> logging.Logger:
    """Create (or retrieve) the shared 'medical_vqa' logger.

    The logger writes to both the console and a file under `log_dir`.

    Args:
        log_dir:  Directory where log files are stored. Created if missing.
        log_name: Base name for the log file (without extension).
                  Defaults to 'run_YYYYMMDD_HHMMSS'.

                  TIP — pass the SAME log_name to train.py and test.py so that
                  training and test results are appended to one shared log file:

                      python train.py --log_name pathvqa_run1 ...
                      python test.py  --log_name pathvqa_run1 ...
                      # → logs/pathvqa_run1.log contains both

    Returns:
        A configured logging.Logger instance (singleton per process).
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_name is None:
        log_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_file = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger("medical_vqa")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File  (mode="a" → append, so train + test go into the same file)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file : {os.path.abspath(log_file)}")
    return logger
