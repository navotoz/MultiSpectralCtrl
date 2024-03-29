import logging
import threading as th
from time import sleep

import utils.constants as const
from devices.FilterWheel import FilterWheelAbstract
from devices.FilterWheel.DummyFilterWheel import FilterWheel as DummyFilterWheel
from devices.FilterWheel.FilterWheel import FilterWheel


class FilterWheelCtrl:
    def __init__(self, logging_handlers: (tuple, list)):
        super().__init__()
        self._logging_handlers = logging_handlers
        self._filterwheel_type = const.DEVICE_DUMMY
        self._filterwheel: (FilterWheelAbstract, None) = DummyFilterWheel()
        self._position_names_dict = {}
        self._reversed_positions_names_dict = {}
        self._lock_access = th.Lock()
        self._th_connect = th.Thread(target=self._th_connect_function, name='connect_filterwheel', daemon=True)
        self._th_connect.start()

    def _th_connect_function(self):
        while True:
            with self._lock_access:  # lock should be inside function, to prevent idle locked waiting
                if self._filterwheel_type == const.DEVICE_DUMMY:
                    try:
                        filterwheel = FilterWheel(logging_handlers=self._logging_handlers)
                        self._filterwheel_type = const.DEVICE_REAL
                        self._filterwheel = filterwheel
                        return
                    except (RuntimeError, BrokenPipeError, ValueError, IndexError):
                        pass
                elif self._filterwheel_type == const.DEVICE_OFF:
                    return
            sleep(1)

    def __del__(self):
        try:
            self._filterwheel_type = const.DEVICE_OFF  # to kill check_dummy thread
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        try:
            del self._filterwheel
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

    @property
    def position(self):
        with self._lock_access:
            return self._filterwheel.position

    @position.setter
    def position(self, next_position):
        with self._lock_access:
            self._filterwheel.position = next_position

    @property
    def position_count(self):
        with self._lock_access:
            return self._filterwheel.position_count

    @property
    def is_dummy(self):
        with self._lock_access:
            return self._filterwheel_type == const.DEVICE_DUMMY

    @property
    def log(self) -> logging.Logger:
        with self._lock_access:
            return self._filterwheel.log

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
            self.log.error(msg)
            raise ValueError(msg)
        if list(range(1, positions_count + 1)) != sorted_keys:
            msg = f'The given names keys does not have all the positions. ' \
                  f'Expected {self.position_count} and got {len(names_dict)}.'
            self.log.error(msg)
            raise ValueError(msg)
        self._position_names_dict = names_dict.copy()  # create numbers to names dict
        if len(set(names_dict.values())) == len(names_dict.values()):
            reversed_generator = (reversed(item) for item in names_dict.copy().items())
            self._reversed_positions_names_dict = {key: val for key, val in reversed_generator}
        else:
            msg = f'There are duplicates in the given position names dict {names_dict}.'
            self.log.error(msg)
            raise ValueError(msg)
        self.log.debug(f'Changed positions name dict to {list(self._reversed_positions_names_dict.keys())}.')
