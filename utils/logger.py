import logging
from pathlib import Path


def make_logging_handlers(logfile_path: (None, Path) = None, verbose: bool = False, logging_level: int = logging.INFO):
    fmt = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    handlers_list = []
    handlers_list.append(logging.StreamHandler()) if verbose else None
    handlers_list.append(logging.FileHandler(str(logfile_path.resolve()), mode='w')) if logfile_path else None
    for handler in handlers_list:
        handler.setFormatter(fmt)
        handler.setLevel(logging_level)
    return handlers_list


def make_logger(name: str, handlers: (list, tuple, logging.Handler), level: (int, None) = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not handlers or len(handlers) == 0:
        handlers = make_logging_handlers(logfile_path=None, verbose=True)
    logger.addHandler(*handlers)
    logger.setLevel(handlers[0].level if not level else level)
    return logger
