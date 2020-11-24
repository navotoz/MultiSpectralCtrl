import logging
from pathlib import Path


class DashLogger(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.logs = list()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            self.logs = self.logs[-1000:]
            self.flush()
        except Exception:
            self.handleError(record)


def make_device_logging_handler(name, logging_handlers):
    handler = [handler for handler in logging_handlers if isinstance(handler, logging.FileHandler)]
    path = Path('log') / f'{name.lower()}.txt'
    if handler:
        path = Path(handler[0].baseFilename).parent / path
    else:
        path = Path().cwd() / path
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)
    handler = logging.FileHandler(str(path), mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(make_fmt())
    return logging_handlers + (handler,)


def make_fmt():
    return logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt='%Y-%m-%d %H:%M:%S')


def make_logging_handlers(logfile_path: (None, Path) = None, verbose: bool = False) -> tuple:
    fmt = make_fmt()
    handlers_list = []
    if verbose:
        handlers_list.append(logging.StreamHandler())
        handlers_list[0].name = 'stdout'
    if logfile_path and not logfile_path.parent.is_dir():
        logfile_path.parent.mkdir(parents=True)
    handlers_list.append(logging.FileHandler(str(logfile_path), mode='w')) if logfile_path else None
    for handler in handlers_list:
        handler.setFormatter(fmt)
    handlers_list.append(dash_logger)
    return tuple(handlers_list)


def make_logger(name: str, handlers: (list, tuple), level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    for idx in range(len(handlers)):
        if handlers[idx].name == 'stdout':
            handlers[idx].setLevel(level)
        else:
            handlers[idx].setLevel(logging.DEBUG)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class ExceptionsLogger:
    def __init__(self):
        self._logger = logging.getLogger('ExceptionsLogger')
        path = Path('log') / 'critical.txt'
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)
        self._logger.addHandler(logging.FileHandler(path, mode='w'))
        self._logger.addHandler(logging.StreamHandler())

    def flush(self):
        pass

    def write(self, message):
        if message != '\n':
            self._logger.critical(message.split('\n')[0])


dash_logger = DashLogger()
dash_logger.setFormatter(make_fmt())
dash_logger.setLevel(logging.DEBUG)
