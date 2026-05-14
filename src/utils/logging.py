from __future__ import annotations
from pathlib import Path
import logging

def setup_logger(path: str|Path):
    logger=logging.getLogger('sheaf_mhd_experiment'); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    Path(path).parent.mkdir(parents=True, exist_ok=True); fh=logging.FileHandler(path); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
