import multiprocessing as mp
import threading as th
from abc import abstractmethod

from utils.tools import SyncFlag


class DeviceAbstract(mp.Process):
    _workers_dict = {}

    def __init__(self, event_stop: mp.Event, logging_handlers: (tuple, list)):
        super().__init__()
        self._event_stop: mp.Event = event_stop
        self._flag_run = SyncFlag(init_state=True)
        self._logging_handlers = logging_handlers

    def run(self):
        self._workers_dict['event_stop'] = th.Thread(target=self._th_stopper, name='event_stop', daemon=False)
        self._workers_dict['event_stop'].start()

        self._run()

    @abstractmethod
    def _run(self):
        raise NotImplementedError

    def _th_stopper(self):
        self._event_stop.wait()
        self.terminate()

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
        if hasattr(self, '_event_stop') and isinstance(self._event_stop, mp.synchronize.Event):
            self._event_stop.set()
        if hasattr(self, '_flag_run') and isinstance(self._flag_run, SyncFlag):
            self._flag_run.set(False)
        self.terminate()
