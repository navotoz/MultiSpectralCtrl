from abc import abstractmethod
from logging import Logger

# constants
DEFAULT_FILTER_NAMES_DICT = {1: '0', 2: '480', 3: '520', 4: '550', 5: '670', 6: '700'}  # 0 is glass
FILTERWHEEL_RECV_WAIT_TIME_IN_SEC = 0.2  # seconds


# FilterWheel command encodings for serial connection
def SET_POSITION(n):
    return f'pos={n}'.encode('utf-8')


def SET_SPEED_MODE(mode):
    return f'speed={mode}'.encode('utf-8')


GET_ID = b'*idn?'
GET_POSITION = b'pos?'
GET_SPEED_MODE = b'speed?'
SET_SENSOR_MODE = b'sensors=0'
GET_SENSOR_MODE = b'sensors?'
GET_POSITION_COUNT = b'pcount?'


class FilterWheelAbstract:
    _reversed_positions_names_dict = _position_names_dict = None

    def __init__(self, connection, logger: Logger):
        self._log = logger
        self._conn = connection
        self.position_names_dict = DEFAULT_FILTER_NAMES_DICT

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        if self._conn:
            self._conn.close()
        self._log.info("Disconnecting.")

    @property
    @abstractmethod
    def is_dummy(self) -> bool:
        pass

    @property
    @abstractmethod
    def position(self) -> dict:
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def speed(self):
        pass

    @property
    @abstractmethod
    def position_count(self):
        pass

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
            self._log.error(msg)
            raise ValueError(msg)
        if list(range(1, positions_count + 1)) != sorted_keys:
            msg = f'The given names keys does not have all the positions. ' \
                  f'Expected {self.position_count} and got {len(names_dict)}.'
            self._log.error(msg)
            raise ValueError(msg)
        self._position_names_dict = names_dict.copy()  # create numbers to names dict
        if len(set(names_dict.values())) == len(names_dict.values()):
            reversed_generator = (reversed(item) for item in names_dict.copy().items())
            self._reversed_positions_names_dict = {key: val for key, val in reversed_generator}
        else:
            msg = f'There are duplicates in the given position names dict {names_dict}.'
            self._log.error(msg)
            raise ValueError(msg)
        self._log.debug(f'Changed positions name dict to {list(self._reversed_positions_names_dict.keys())}.')

    def is_position_name_valid(self, name: str) -> bool:
        """
        Checks if the given name is indeed in the position dict.
        Args:
            name: a string of the name of a filter.
        Returns:
            True if the name is in the position dict or False if the filter name is not in the dictionary.
        """
        return name in self.position_names_dict.values()

    def get_position_from_name(self, name: str) -> int:
        """
        Returns the position of the input on the FilterWheel if valid, else -1.

        Args:
            name a string with a filter name
        Returns:
            The position of the given name on the FilterWheel if a valid, else -1.
        """
        if not self.is_position_name_valid(name):
            print(self._log)
            self._log.warning(f"Given position name {name} not in position names dict.")
            return -1
        return self._reversed_positions_names_dict[name]

    @property
    def log(self):
        return self._log