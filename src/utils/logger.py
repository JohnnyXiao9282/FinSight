import logging
import sys
from pathlib import Path
from datetime import datetime

# Default log directory
LOG_DIR = Path(__file__).parent.parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Always add file handler to logs/finsight.log
    if log_file is None:
        log_file = LOG_DIR / 'finsight.log'
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str):
    return logging.getLogger(name)
