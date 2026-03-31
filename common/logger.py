"""
Unified logging for all tasks.

Usage:
    from common.logger import get_logger
    logger = get_logger("task1", log_dir="outputs/task1/exp01")
"""

import logging
import sys
from pathlib import Path


def get_logger(name, log_dir=None, log_file="train.log", level=logging.INFO):
    """
    Create a logger that writes to console and optionally to a file.

    Args:
        name: Logger name (e.g., "task1.baseline")
        log_dir: If provided, also write to {log_dir}/{log_file}
        log_file: Log filename (default: "train.log")
        level: Logging level
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
