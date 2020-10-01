import logging
from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
import random

class DummyFilterWheel:
    _reversed_pos_names_dict = None
    __curr_position = 1
    __speed = 1

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.__log.info("Disconnecting from dummy FilterWheel.")

    def __init__(self, model_name: str = 'FW102C', logger: (None, logging.Logger) = None):
        self.__log = logging.getLogger('DummyFilterWheel') if not logger else logger

        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy FilterWheel simulates failure.')

        # set default options
        _ = self.id  # sometimes the serial buffer holds a CMD_NOT_DEFINED, so this cmd is to clear the buffer.
        positions_count = self.position_count
        self.is_position_in_limits = lambda position: 0 < position <= positions_count
        self._position_names_dict = dict(zip(range(1, positions_count + 1),
                                             list(map(lambda x: str(x), range(1, positions_count + 1)))))

    @property
    def position(self) -> dict:
        """
        The position of the FilterWheel as a dictionary with (number, name) as keys.

        Returns:
            A dictionary with (number, name) for FilterWheel current position.
        """
        pos_number = self.__curr_position
        pos_name = self._position_names_dict[pos_number]
        self.__log.debug(f"PosNum{pos_number}_PosName{pos_name}.")
        return dict(number=pos_number, name=pos_name)

    @position.setter
    def position(self, next_position: (int, str)):
        """
        Sets the position for the FilterWheel.
        If given position is illegal, logs as warning.

        Args:
            next_position: the next position in the FilterWheel as either a number or the name of the filter.
        """
        next_position = self.get_position_from_name(next_position) if isinstance(next_position, str) else next_position
        if self.is_position_in_limits(next_position):
            self.__curr_position = next_position
        else:
            self.__log.warning(f'Position {next_position} is invalid.')

    @property
    def id(self) -> str:
        """
        Returns:
            The id of the FilterWheel as a str.
        """
        return 'THORLABS FW102C/FW212C Filter Wheel version 1.07'

    @property
    def speed(self) -> int:
        return self.__speed

    @speed.setter
    def speed(self, mode: str):
        self.__speed = 0 if mode == 'slow' else 1

    @property
    def position_count(self) -> int:
        """
        Returns: how many positions are valid for the FilterWheel.
        """
        return 6

    @property
    def position_names_dict(self) -> dict:
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
            self.__log.error(msg)
            raise ValueError(msg)
        if list(range(1, positions_count + 1)) != sorted_keys:
            msg = f'The given names keys does not have all the positions. ' \
                  f'Expected {self.position_count} and got {len(names_dict)}.'
            self.__log.error(msg)
            raise ValueError(msg)
        self._position_names_dict = names_dict.copy()  # create numbers to names dict
        if len(set(names_dict.values())) == len(names_dict.values()):
            reversed_generator = (reversed(item) for item in names_dict.copy().items())
            self._reversed_pos_names_dict = {key: val for key, val in reversed_generator}
        else:
            msg = f'There are duplicates in the given position names dict {names_dict}.'
            self.__log.error(msg)
            raise ValueError(msg)
        self.__log.debug(f'Changed positions name dict to {list(self._reversed_pos_names_dict.keys())}.')

    def is_position_name_valid(self, name: str) -> bool:
        return name in self.position_names_dict.values()

    def get_position_from_name(self, name: str) -> int:
        if not self.is_position_name_valid(name):
            self.__log.warning(f"Given position name {name} not in position names dict.")
            return -1
        return self._reversed_pos_names_dict[name]
