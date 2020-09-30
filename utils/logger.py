import logging
from pathlib import Path


def make_logging_handlers(logfile_path: (None, Path) = None, verbose: bool = False):
    fmt = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    handlers_list = []
    handlers_list.append(logging.StreamHandler()) if verbose else None
    handlers_list.append(logging.FileHandler(str(logfile_path.resolve()), mode='w')) if logfile_path else None
    for handler in handlers_list:
        handler.setFormatter(fmt)
    return handlers_list


def make_logger(name: str, handlers: (list, tuple, logging.Handler), level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(*handlers)
    logger.setLevel(level)
    return logger
