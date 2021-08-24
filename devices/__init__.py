import multiprocessing as mp
import threading as th
from abc import abstractmethod

from utils.tools import SyncFlag


class DeviceAbstract(mp.Process):
    _workers_dict = {}
    _flags_pipes_list = []

    def __init__(self, event_stop: mp.Event, logging_handlers: (tuple, list)):
        super().__init__()
        self._event_stop: mp.Event = event_stop
        self._flag_run = SyncFlag(init_state=True)
        self._logging_handlers = logging_handlers

    def run(self):
        self._workers_dict['event_stop'] = th.Thread(target=self._th_stopper, name='event_stop', daemon=False)
        self._workers_dict['event_stop'].start()

        self._workers_dict['cmd_parser'] = th.Thread(target=self._th_cmd_parser, name='cmd_parser', daemon=True)
        self._workers_dict['cmd_parser'].start()

        self._run()

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def _th_cmd_parser(self):
        pass

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
        if hasattr(self, '_flag_run'):
            self._flag_run.set(False)
        for p in self._flags_pipes_list:
            try:
                p.set(False)
            except (RuntimeError, AssertionError, AttributeError, TypeError):
                pass
        self._terminate_device_specifics()
        self._wait_for_threads_to_exit()
        try:
            self.kill()
        except (AttributeError, AssertionError, TypeError, KeyError):
            pass

    @abstractmethod
    def _terminate_device_specifics(self):
        pass

    def __del__(self):
        if hasattr(self, '_event_stop') and isinstance(self._event_stop, mp.synchronize.Event):
            self._event_stop.set()
        self.terminate()
