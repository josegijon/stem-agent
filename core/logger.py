"""
logger.py

Shared logging configuration for the stem agent system.

All modules import their logger from here to ensure consistent
formatting, levels, and output across the entire pipeline.

Design decisions:
- propagate=False: prevents duplicate log entries when the root logger
  also has handlers configured.
- LOG_LEVEL from environment: allows verbosity control without code changes.
- Single handler per logger: guard against duplicate handlers if get_logger
  is called multiple times for the same module name.

Possible extension: add run_id and domain context via LoggerAdapter
for correlating logs across parallel or sequential runs.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Starting phase...")
"""

import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path("logs")


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Uses the LOG_LEVEL environment variable to set verbosity.
    Defaults to INFO if not set.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A configured Logger instance with propagation disabled
        to prevent duplicate log entries.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if debug_mode:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
