from numbers import Number
import multiprocessing as mp
import threading as th
from pathlib import Path
from time import sleep

import utils.constants as const
from devices import DeviceAbstract
from utils.logger import make_logging_handlers, make_logger, make_device_logging_handler
from utils.tools import wait_for_time, DuplexPipe
from devices.FilterWheel.FilterWheel import FilterWheel
from devices.FilterWheel.DummyFilterWheel import FilterWheel as DummyFilterWheel


class FilterWheelProc(DeviceAbstract):
    def _terminate_device_specifics(self):
        pass

    def __init__(self,
                 logging_handlers: (tuple, list),
                 event_stop: mp.Event,
                 cmd_pipe: DuplexPipe):
        super(FilterWheelProc, self).__init__(event_stop, logging_handlers, None)
        self._logging_handlers = make_device_logging_handler(f'{const.FILTERWHEEL_NAME}', logging_handlers)
        self._cmd_pipe = cmd_pipe
        self._flags_pipes_list = [self._cmd_pipe.flag_run]
        self._filterwheel_type = const.DEVICE_DUMMY
        self._filterwheel: (FilterWheel, DummyFilterWheel, None) = DummyFilterWheel()
        self._position_names_dict = {}
        self._reversed_positions_names_dict = {}

        self._lock_access = th.Lock()

    def _run(self):
        self._filterwheel = DummyFilterWheel(logging_handlers=self._logging_handlers)
        self._filterwheel_type = const.DEVICE_DUMMY

        self._workers_dict['check_dummy'] = th.Thread(target=self._th_check_dummy, name='check_dummy')
        self._workers_dict['check_dummy'].start()

    def _th_check_dummy(self):
        def get():
            with self._lock_access:  # lock should be inside function, to prevent idle locked waiting
                # try to change to real blackbody
                if self._filterwheel_type == const.DEVICE_DUMMY:
                    try:
                        filterwheel = FilterWheel(logging_handlers=self._logging_handlers)
                        self._filterwheel_type = const.DEVICE_REAL
                        self._filterwheel = filterwheel
                    except (RuntimeError, BrokenPipeError):
                        pass

                # make sure the blackbody is still connected
                elif self._filterwheel_type == const.DEVICE_REAL:
                    try:
                        self._filterwheel.echo()
                    except (RuntimeError, BrokenPipeError):
                        if self._filterwheel_type == const.DEVICE_REAL:  # echo fail, change from real to dummy
                            self._filterwheel = DummyFilterWheel(logging_handlers=self._logging_handlers)
                            self._filterwheel_type = const.DEVICE_DUMMY

        getter = wait_for_time(get, 1)
        while self._flag_run:
            getter()

    def _th_cmd_parser(self):
        while self._flag_run:
            if (cmd := self._cmd_pipe.recv()) is not None:
                cmd, value = cmd
                if cmd == const.FILTERWHEEL_NAME:
                    with self._lock_access:
                        if value is True:
                            self._cmd_pipe.send(self._filterwheel_type)
                        elif value != self._filterwheel_type:
                            self._filterwheel = None
                            self._filterwheel_type = const.DEVICE_DUMMY
                            try:
                                if value == const.DEVICE_REAL:
                                    self._filterwheel = FilterWheel(logging_handlers=self._logging_handlers)
                                elif value == const.DEVICE_DUMMY:
                                    self._filterwheel = DummyFilterWheel(logging_handlers=self._logging_handlers)
                                self._filterwheel_type = value
                            except RuntimeError:
                                self._filterwheel = DummyFilterWheel(logging_handlers=self._logging_handlers)
                        else:
                            self._filterwheel_type = value
                    self._cmd_pipe.send(self._filterwheel_type)
                elif cmd == const.FILTERWHEEL_POSITION:
                    # todo: make setter and getter here
                    with self._lock_access:
                        if isinstance(value, Number) and not isinstance(value, bool):
                            self._filterwheel.temperature = value
                            self._temperature = self._filterwheel.temperature
                        self._cmd_pipe.send(self._temperature)
                else:
                    self._cmd_pipe.send(None)

    @property
    def position_count(self):
        return self._filterwheel.position_count

    @property
    def is_dummy(self):
        return self._filterwheel_type == const.DEVICE_DUMMY

    @property
    def position_names_dict(self):
        return self._position_names_dict

    @position_names_dict.setter
    def position_names_dict(self, names_dict: dict):
        """
        Sets the positions names. Also creates a reverse-dictionary with names a keys.

        Args:
            names_dict: a dictionary of names for the positions.
        """
        positions_count = self.position_count
        sorted_keys = list(names_dict.keys())
        sorted_keys.sort()
        if len(sorted_keys) < positions_count:
            msg = f'Not enough keys in given names dict {names_dict}.'
            self._filterwheel._log.error(msg)
            raise ValueError(msg)
        if list(range(1, positions_count + 1)) != sorted_keys:
            msg = f'The given names keys does not have all the positions. ' \
                  f'Expected {self.position_count} and got {len(names_dict)}.'
            self._filterwheel._log.error(msg)
            raise ValueError(msg)
        self._position_names_dict = names_dict.copy()  # create numbers to names dict
        if len(set(names_dict.values())) == len(names_dict.values()):
            reversed_generator = (reversed(item) for item in names_dict.copy().items())
            self._reversed_positions_names_dict = {key: val for key, val in reversed_generator}
        else:
            msg = f'There are duplicates in the given position names dict {names_dict}.'
            self._filterwheel._log.error(msg)
            raise ValueError(msg)
        self._filterwheel._log.debug(f'Changed positions name dict to {list(self._reversed_positions_names_dict.keys())}.')
