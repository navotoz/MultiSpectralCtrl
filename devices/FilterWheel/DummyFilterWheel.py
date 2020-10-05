from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
import random
from utils.logger import make_logger
from devices.FilterWheel import FilterWheelAbstract


class FilterWheel(FilterWheelAbstract):
    __dummy_curr_position = 1
    __dummy_speed = 1

    def __init__(self, model_name: str = 'FW102C', logging_handlers: tuple = ()):
        self._log = make_logger('DummyFilterWheel', logging_handlers)

        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy FilterWheel simulates failure.')

        # set default options
        positions_count = self.position_count
        self.is_position_in_limits = lambda position: 0 < position <= positions_count
        super().__init__(None, self._log)
        self._log.info("Using DummyFilterWheel")

    @property
    def is_dummy(self)->bool:
        return True

    @property
    def position(self) -> dict:
        """
        The position of the FilterWheel as a dictionary with (number, name) as keys.

        Returns:
            A dictionary with (number, name) for FilterWheel current position.
        """
        pos_number = self.__dummy_curr_position
        pos_name = self.__position_names_dict[pos_number]
        self._log.debug(f"PosNum{pos_number}_PosName{pos_name}.")
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
            self.__dummy_curr_position = next_position
        else:
            self._log.warning(f'Position {next_position} is invalid.')

    @property
    def id(self) -> str:
        """
        Returns:
            The id of the FilterWheel as a str.
        """
        return 'THORLABS FW102C/FW212C Filter Wheel version 1.07'

    @property
    def speed(self) -> int:
        return self.__dummy_speed

    @speed.setter
    def speed(self, mode: str):
        self.__dummy_speed = 0 if mode == 'slow' else 1

    @property
    def position_count(self) -> int:
        """
        Returns: how many positions are valid for the FilterWheel.
        """
        return 6
