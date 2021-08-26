import multiprocessing as mp
from abc import abstractmethod

from utils.tools import SyncFlag


class DeviceAbstract(mp.Process):
    _workers_dict = {}

    def __init__(self):
        super().__init__()
        self._flag_run = SyncFlag(init_state=True)

    def run(self):
        self._run()

    @abstractmethod
    def _run(self):
        raise NotImplementedError

    def _wait_for_threads_to_exit(self):
        for key, t in self._workers_dict.items():
            if t.daemon:
                continue
            try:
                t.join()
            except (RuntimeError, AssertionError, AttributeError):
                pass

    def terminate(self) -> None:
        if hasattr(self, '_flag_run') and isinstance(self._flag_run, SyncFlag):
            self._flag_run.set(False)
        self._terminate_device_specifics()
        self._wait_for_threads_to_exit()
        try:
            self.kill()
        except (AttributeError, AssertionError, TypeError, KeyError):
            pass

    @abstractmethod
    def _terminate_device_specifics(self):
        raise NotImplementedError

    def __del__(self):
        if hasattr(self, '_flag_run') and isinstance(self._flag_run, SyncFlag):
            self._flag_run.set(False)
        self.terminate()
