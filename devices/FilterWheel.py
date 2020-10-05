from time import sleep
import serial.tools.list_ports
import serial
import logging
from utils.constants import RECV_WAIT_TIME_IN_SEC
from utils.constants import GET_ID, GET_POSITION, GET_SPEED_MODE, SET_SENSOR_MODE, GET_SENSOR_MODE, GET_POSITION_COUNT
from utils.logger import make_logger, make_logging_handlers
from devices import FilterWheelAbstract


def SET_POSITION(n):
    return f'pos={n}'.encode('utf-8')


def SET_SPEED_MODE(mode):
    return f'speed={mode}'.encode('utf-8')


class FilterWheel(FilterWheelAbstract):
    __reversed_pos_names_dict = None

    def __init__(self, model_name: str = 'FW102C', logging_handlers: tuple = ()):
        super().__init__()
        self.__log = make_logger('FilterWheel', logging_handlers)

        port = [p for p in serial.tools.list_ports.comports() if model_name in p]
        if len(port) == 0:
            self.__log.error(f"FilterWheel {model_name} not detected.")
            raise RuntimeError('This model was not detected.')
        port = port[0]  # only one port should remain
        try:
            self.__conn = serial.Serial(port.device, baudrate=115200,
                                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
        except (serial.SerialException, RuntimeError) as err:
            self.__log.critical(err)
            raise RuntimeError(err)

        if self.__conn.is_open:
            self.__log.info("Connected to FilterWheel at {}.".format(port))
            self.__conn.flushInput()
            self.__conn.flushOutput()
            self.__conn.timeout = 3
        else:
            self.__log.critical("Couldn't connect to FilterWheel!")
            raise RuntimeError("Couldn't connect to FilterWheel!")

        # set default options
        _ = self.id  # sometimes the serial buffer holds a CMD_NOT_DEFINED, so this cmd is to clear the buffer.
        positions_count = self.position_count
        self.is_position_in_limits = lambda position: 0 < position <= positions_count
        self.__position_names_dict = dict(zip(range(1, positions_count + 1),
                                              list(map(lambda x: str(x), range(1, positions_count + 1)))))
        self.__set_sensor_mode_to_off()

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        if self.__conn is not None:
            self.__conn.close()
        self.__log.info("Disconnecting from FilterWheel.")

    def __send(self, command: bytes):
        self.__conn.write(command + b'\r')

    def __recv(self, min_bytes: int = 0, blocking: bool = True) -> list:
        sleep(RECV_WAIT_TIME_IN_SEC)
        num_of_bytes = self.__conn.inWaiting()
        if blocking:
            while num_of_bytes < min_bytes:
                num_of_bytes = self.__conn.inWaiting()
        return self.__conn.read(num_of_bytes).decode().split('\r')

    def __send_and_recv(self, cmd: bytes) -> list:
        """
        Send a command and waits for the answer.
        Attempts to receive the answer a second time if the first attempt fails.

        Args:
            cmd: bytes of the requested command.

        Returns:
            A list of the return values for the given command.
        """
        self.__send(cmd)
        ret_list = self.__recv()
        if len(ret_list) < 3:
            ret_list = self.__recv()
        return ret_list

    def __set_sensor_mode_to_off(self):
        """
        Sets the sensor mode in the FilterWheel to 0,
        meaning that the sensors turn off then the wheel is idle to eliminate stray light
        """
        if int(self.__send_and_recv(GET_SENSOR_MODE)[1]) != 0:
            self.__send_and_recv(SET_SENSOR_MODE)

    @property
    def is_dummy(self) -> bool:
        return False

    @property
    def position(self) -> dict:
        """
        The position of the FilterWheel as a dictionary with (number, name) as keys.

        Returns:
            A dictionary with (number, name) for FilterWheel current position.
        """
        pos_number = self.__send_and_recv(GET_POSITION)
        while len(pos_number) < 2:  # busy-waiting for answer
            pos_number = self.__send_and_recv(GET_POSITION)
        pos_number = int(pos_number[1])
        pos_name = self.__position_names_dict[pos_number]
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
            curr_position = 0
            while curr_position != next_position:  # busy-waiting until position stabilizes
                self.__send_and_recv(SET_POSITION(next_position))
                curr_position = self.position['number']
        else:
            self.__log.warning(f'Position {next_position} is invalid.')

    @property
    def id(self) -> str:
        """
        Returns:
            The id of the FilterWheel as a str.
        """
        id_ = self.__send_and_recv(GET_ID)[1]
        self.__log.debug(id_)
        return id_

    @property
    def speed(self) -> int:
        """
        The SpeedMode of the FilterWheel. Either "Fast" or "Slow". Defaults to "Fast".
        """
        speed = int(self.__send_and_recv(GET_SPEED_MODE)[1])
        self.__log.debug("SpeedMode" + ("Fast" if speed == 1 else "Slow") + ".")
        return speed

    @speed.setter
    def speed(self, mode: str):
        """
        Setter for the SpeedMode of the FilterWheel. Expects a str of either "Fast" of "Slow".

        Args:
            mode: str with either "fast" or "slow".
        """
        mode = mode.lower()
        if mode != 'fast' and mode != 'slow':
            err_msg = f"Speed mode {mode} not implemented. Try either 'fast' or 'slow'."
            self.__log.error(err_msg)
            raise ValueError(err_msg)
        self.__send_and_recv(SET_SPEED_MODE(1 if mode == 'fast' else 0))

    @property
    def position_count(self) -> int:
        """
        Returns: how many positions are valid for the FilterWheel.
        """
        return int(self.__send_and_recv(GET_POSITION_COUNT)[1])

    @property
    def position_names_dict(self) -> dict:
        """
        A dictionary where the keys are the position of a filter as integer and the value is the filter's name.
        """
        return self.__position_names_dict

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
        self.__position_names_dict = names_dict.copy()  # create numbers to names dict
        if len(set(names_dict.values())) == len(names_dict.values()):
            reversed_generator = (reversed(item) for item in names_dict.copy().items())
            self.__reversed_pos_names_dict = {key: val for key, val in reversed_generator}
        else:
            msg = f'There are duplicates in the given position names dict {names_dict}.'
            self.__log.error(msg)
            raise ValueError(msg)
        self.__log.debug(f'Changed positions name dict to {list(self.__reversed_pos_names_dict.keys())}.')
