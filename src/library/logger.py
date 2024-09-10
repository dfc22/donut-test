import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.library.env import ENV


def get_log_level(level_name: str) -> int:
    return logging.getLevelNamesMapping()[level_name]


LOG_LEVEL = get_log_level(ENV.get("LOG_LEVEL", "INFO"))
LOG_DIR = Path(ENV.get("LOG_DIR", "./logs"))
TIMEZONE = ZoneInfo(ENV.get("TZ", "UTC"))


def configure_root_logger() -> None:
    formatter = logging.Formatter(
        fmt="[%(asctime)s]:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S(%Z)",
    )
    now = datetime.now(TIMEZONE).strftime("%Y-%m-%d_%Hh%Mm%Ss")
    log_file = LOG_DIR / f"{sys.argv[0]}_{now}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    handlers = [stream_handler, file_handler]
    for handler in handlers:
        handler.setLevel(LOG_LEVEL)
        handler.setFormatter(formatter)
    logging.basicConfig(level=LOG_LEVEL, handlers=handlers)
    logging.info("log file is %s", log_file)
    logging.info("log level is %s", LOG_LEVEL)
    logging.info("timezone is %s", TIMEZONE)


class Logger(logging.getLoggerClass()):
    _is_instanced = False

    def __new__(cls) -> logging.Logger:
        if not cls._is_instanced:
            cls._is_instanced = True
            configure_root_logger()
        return logging.getLogger(__name__)


LOGGER = Logger()
