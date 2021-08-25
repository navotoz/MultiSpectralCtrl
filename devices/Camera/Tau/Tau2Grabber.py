import logging
import struct
from pathlib import Path

import numpy as np
import yaml
from pyftdi.ftdi import Ftdi, FtdiError

import devices.Camera.Tau.tau2_config as ptc
from devices.Camera.Tau.TauCameraCtrl import Tau, _make_packet
from devices.Camera.utils import connect_ftdi, _is_8bit_image_borders_valid, BytesBuffer, HEADER_SIZE_IN_BYTES, \
    generate_subsets_indices_in_string, parse_incoming_message
from utils.logger import make_logger, make_logging_handlers
import threading as th

from utils.tools import SyncFlag

KELVIN2CELSIUS = 273.15
BORDER_VALUE = 64
FTDI_PACKET_SIZE = 512 * 8
SYNC_MSG = b'SYNC' + struct.pack(4 * 'B', *[0, 0, 0, 0])


class Tau2Grabber(Tau):
    def _write(self, data: bytes) -> None:
        buffer = b"UART"
        buffer += int(len(data)).to_bytes(1, byteorder='big')  # doesn't matter
        buffer += data
        try:
            with self._lock_access_ftdi:
                self._ftdi.write_data(buffer)
            self._log.debug(f"Send {data}")
        except FtdiError:
            self._log.debug('Write error.')
            self._reset()

    def set_params_by_dict(self, yaml_or_dict: (Path, dict)):
        if isinstance(yaml_or_dict, Path):
            params = yaml.safe_load(yaml_or_dict)
        else:
            params = yaml_or_dict.copy()
        self.n_retry, default_n_retries = 10, self.n_retry
        self.ffc_mode = params.get('ffc_mode', 'external')
        self.isotherm = params.get('isotherm', 0)
        self.dde = params.get('dde', 0)
        self.tlinear = params.get('tlinear', 0)
        self.gain = params.get('gain', 'high')
        self.agc = params.get('agc', 'manual')
        self.sso = params.get('sso', 0)
        self.contrast = params.get('contrast', 0)
        self.brightness = params.get('brightness', 0)
        self.brightness_bias = params.get('brightness_bias', 0)
        self.cmos_depth = params.get('cmos_depth', 0)  # 14bit pre AGC
        self.fps = params.get('fps', 4)  # 60Hz NTSC
        # self.correction_mask = params.get('corr_mask', 0)  # off
        self.n_retry = default_n_retries

    @property
    def n_retry(self) -> int:
        return self._n_retry

    @n_retry.setter
    def n_retry(self, n_retry: int):
        self._n_retry = n_retry

    def __init__(self, vid=0x0403, pid=0x6010,
                 logging_handlers: tuple = make_logging_handlers(None, True),
                 logging_level: int = logging.INFO):
        logger = make_logger('TeaxGrabber', logging_handlers, logging_level)
        try:
            super().__init__(logger=logger)
        except IOError:
            pass
        try:
            self._ftdi = connect_ftdi(vid, pid)
        except RuntimeError:
            raise RuntimeError('Could not connect to the Tau2 camera.')
        self._lock_access_ftdi = th.Lock()
        self._lock_parse_command = th.Lock()
        self._event_allow_all_commands = th.Event()
        self._event_allow_all_commands.set()
        self._event_read = th.Event()
        self._event_read.clear()
        self._event_msg_ready = th.Event()
        self._event_msg_ready.clear()

        self._n_retry = 3
        self._width = self.width
        self._height = self.height
        self._frame_size = 2 * self.height * self.width + 6 + 4 * self.height  # 6 byte header, 4 bytes pad per row
        self._flag_run = SyncFlag(init_state=True)

        self._buffer = BytesBuffer(flag_run=self._flag_run, size_to_signal=self._frame_size)

        self._len_command_in_bytes = 0
        self._thread_read = th.Thread(target=self._th_reader_func, name='th_tau2grabber_reader', daemon=True)
        self._thread_read.start()
        self._log.info('Ready.')

        self.ffc_mode = ptc.FFC_MODE_CODE_DICT['external']


    def __del__(self) -> None:
        if hasattr(self, '_flag_run') and isinstance(self._flag_run, SyncFlag):
            self._flag_run.set(False)
        if hasattr(self, '_buffer') and isinstance(self._buffer, BytesBuffer):
            del self._buffer
        if hasattr(self, '_event_read') and isinstance(self._event_read, th.Event):
            self._event_read.set()
        if hasattr(self, '_event_allow_ftdi_access') and isinstance(self._event_allow_all_commands, th.Event):
            self._event_allow_all_commands.set()
        if hasattr(self, '_thread_read') and isinstance(self._thread_read, th.Thread):
            self._thread_read.join()
        if hasattr(self, '_ftdi') and isinstance(self._ftdi, Ftdi):
            self._ftdi.close()

        if hasattr(self, '_log') and isinstance(self._log, logging.Logger):
            try:
                self._log.critical('Exit.')
            except NameError:
                pass

    def _send_and_recv_threaded(self, command: ptc.Code, argument: (bytes, None), n_retry: int = 3) -> (bytes, None):
        data = _make_packet(command, argument)
        return self._io.parse(data=data, command=command, n_retry=n_retry if n_retry != self.n_retry else self.n_retry)



    def grab(self, to_temperature: bool = False, n_retries: int = 3) -> (np.ndarray, None):
        # Note that in TeAx's official driver, they use a threaded loop
        # to read data as it streams from the camera and they simply
        # process images/commands as they come back. There isn't the same
        # sort of query/response structure that you'd normally see with
        # a serial device as the camera basically vomits data as soon as
        # the port opens.
        #
        # The current approach here aims to allow a more structured way of
        # interacting with the camera by synchronising with the stream whenever
        # some particular data is requested. However in the future it may be better
        # if this is moved to a threaded function that continually services the
        # serial stream and we have some kind of helper function which responds
        # to commands and waits to see the answer from the camera.
        for _ in range(max(1, n_retries)):
            raw_image_8bit = self._io.grab()
            if raw_image_8bit is not None:
                raw_image_16bit = 0x3FFF & np.array(raw_image_8bit).view('uint16')[:, 1:-1]

                if to_temperature:
                    raw_image_16bit = 0.04 * raw_image_16bit - KELVIN2CELSIUS
                return raw_image_16bit
        return None



    def _reset(self) -> None:
        if not self._flag_run:
            return
        self._buffer.clear_buffer()
        self._log.debug('Reset.')

    def _th_reader_func(self) -> None:
        data = None
        while self._flag_run:
            self._event_read.wait()
            with self._lock_access_ftdi:
                try:
                    data = self._ftdi.read_data(FTDI_PACKET_SIZE)
                except FtdiError:
                    pass
            if self._buffer is not None and data is not None:
                if data.rfind(b'UART') >= 0:
                    self._len_command_in_bytes -= 1
                self._buffer += data
            if self._len_command_in_bytes == 0:
                self._event_msg_ready.set()

    def parse(self, data: bytes, command: ptc.Code, n_retry: int) -> (None, bytes):
        self._event_allow_all_commands.wait()
        with self._lock_parse_command:
            self._len_command_in_bytes = command.reply_bytes + HEADER_SIZE_IN_BYTES
            self._buffer.clear_buffer()
            self._event_msg_ready.clear()
            self._event_read.set()
            self._write(data)
            self._event_msg_ready.wait(timeout=2)
            self._event_read.clear()
            parsed_msg = parse_incoming_message(buffer=self._buffer.buffer, command=command)
            if parsed_msg is not None:
                self._log.debug(f"Received {parsed_msg}")
            return parsed_msg

    def grab(self) -> (np.ndarray, None):
        self._event_allow_all_commands.clear()  # only allows this thread to operate
        with self._lock_parse_command:
            for _ in range(max(1, self._n_retries_image)):
                self._event_read.set()
                self._buffer.sync_teax()
                self._buffer.wait_for_size()
                res = self._buffer[:self._frame_size]
                if res and struct.unpack('h', res[6:8])[0] != 0x4000:  # a magic word
                    continue
                frame_width = struct.unpack('h', res[1:3])[0] - 2
                if frame_width != self._width:
                    self._log.debug(f"Received frame has incorrect width of {frame_width}.")
                    continue
                raw_image_8bit = np.frombuffer(res[6:], dtype='uint8').reshape((-1, 2 * (self.width + 2)))
                if not _is_8bit_image_borders_valid(raw_image_8bit, self.height):
                    continue
                self._event_allow_all_commands.set()
                return raw_image_8bit
            self._event_allow_all_commands.set()
            return None
