import logging
import os
from datetime import datetime

MAX_LOG = 100
LOG_DIR = "logs"
LOG_FILENAME_ENV = "MSST_LOG_FILE"


class ColorFormatter(logging.Formatter):
    LEVEL_STYLES = {
        "INFO": "[INFO]    ",
        "DEBUG": "[DEBUG]   ",
        "WARNING": "[WARNING] ",
        "ERROR": "[ERROR]   ",
        "CRITICAL": "[CRITICAL]"
    }

    def format(self, record):
        log_msg = super().format(record)
        if record.levelname in self.LEVEL_STYLES:
            log_msg = log_msg.replace(record.levelname, self.LEVEL_STYLES[record.levelname])
        return log_msg


def set_log_level(logger, level):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
            break


def get_logger(console_level=logging.INFO, max_log=MAX_LOG):
    logger = logging.getLogger("msst_logger")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    formatter = ColorFormatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(console_handler)

    return logger
