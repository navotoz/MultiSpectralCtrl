import logging
from pathlib import Path


def make_logging_handlers(logfile_path: (None, Path) = None, verbose: bool = False) -> tuple:
    fmt = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    handlers_list = []
    handlers_list.append(logging.StreamHandler()) if verbose else None
    handlers_list.append(logging.FileHandler(str(logfile_path), mode='w')) if logfile_path else None
    for handler in handlers_list:
        handler.setFormatter(fmt)
    return tuple(handlers_list)


def make_logger(name: str, handlers: (list, tuple), level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    for idx in range(len(handlers)):
        if not isinstance(handlers[idx], logging.FileHandler):
            handlers[idx].setLevel(level)
        else:
            handlers[idx].setLevel(logging.DEBUG)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
