import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler("app.log", encoding = "utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

"""
from log_config import get_logger

logger = get_logger(__name__)

logger.info("Program started")
logger.warning("Missing data")
logger.error("Failed to load file")
"""