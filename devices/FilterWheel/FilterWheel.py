from time import sleep
import serial.tools.list_ports
import serial
from utils.logger import make_logger, make_device_logging_handler
from devices.FilterWheel import *


class FilterWheel(FilterWheelAbstract):
    def __init__(self, model_name: str = 'FW102C', logging_handlers: tuple = ()):
        logging_handlers = make_device_logging_handler('FilterWheel', logging_handlers)
        self._log = make_logger('FilterWheel', logging_handlers)

        port = [p for p in serial.tools.list_ports.comports() if model_name in p]
        if len(port) == 0:
            self._log.error(f"FilterWheel {model_name} not detected.")
            raise RuntimeError('This model was not detected.')
        port = port[0]  # only one port should remain
        try:
            self._conn = serial.Serial(port.device, baudrate=115200,
                                       parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
        except (serial.SerialException, RuntimeError) as err:
            self._log.critical(err)
            raise RuntimeError(err)

        if self._conn.is_open:
            self._log.info(f"Connected to FilterWheel at {port}.")
            self._conn.flushInput()
            self._conn.flushOutput()
            self._conn.timeout = 3
        else:
            self._log.critical("Couldn't connect to FilterWheel!")
            raise RuntimeError("Couldn't connect to FilterWheel!")

        # set default options
        _ = self.id  # sometimes the serial buffer holds a CMD_NOT_DEFINED, so this cmd is to clear the buffer.
        positions_count = self.position_count
        self.is_position_in_limits = lambda position: 0 < position <= positions_count
        super().__init__(self._conn, self._log)
        self.__set_sensor_mode_to_off()

    def __send(self, command: bytes):
        self._conn.write(command + b'\r')

    def __recv(self, min_bytes: int = 0, blocking: bool = True) -> list:
        sleep(FILTERWHEEL_RECV_WAIT_TIME_IN_SEC)
        num_of_bytes = self._conn.inWaiting()
        if blocking:
            while num_of_bytes < min_bytes:
                num_of_bytes = self._conn.inWaiting()
        return self._conn.read(num_of_bytes).decode().split('\r')

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
        for _ in range(5):
            pos_number = self.__send_and_recv(GET_POSITION)
            while len(pos_number) < 2:  # busy-waiting for answer
                pos_number = self.__send_and_recv(GET_POSITION)
            pos_number = list(filter(lambda x: x.isdecimal(), pos_number))
            if pos_number:
                pos_number = int(pos_number[-1])
            else:
                continue
            pos_name = self._position_names_dict[pos_number]
            return dict(number=pos_number, name=pos_name)
        return dict(number=None, name=None)

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
            curr_position = self.position
            self._log.info(f"Set position {curr_position['number']}# with filter {curr_position['name']}.")
        else:
            self._log.warning(f'Position {next_position} is invalid.')

    @property
    def id(self) -> str:
        """
        Returns:
            The id of the FilterWheel as a str.
        """
        id_ = self.__send_and_recv(GET_ID)[1]
        self._log.debug(id_)
        return id_

    @property
    def speed(self) -> int:
        """
        The SpeedMode of the FilterWheel. Either "Fast" or "Slow". Defaults to "Fast".
        """
        speed = int(self.__send_and_recv(GET_SPEED_MODE)[1])
        self._log.debug("SpeedMode" + ("Fast" if speed == 1 else "Slow") + ".")
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
            self._log.error(err_msg)
            raise ValueError(err_msg)
        self.__send_and_recv(SET_SPEED_MODE(1 if mode == 'fast' else 0))

    @property
    def position_count(self) -> int:
        """
        Returns: how many positions are valid for the FilterWheel.
        """
        return int(self.__send_and_recv(GET_POSITION_COUNT)[1])
