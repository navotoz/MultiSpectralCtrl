import logging
import struct
import threading as th
from pathlib import Path

import numpy as np
import yaml
from pyftdi.ftdi import Ftdi, FtdiError
from usb.core import USBError

import devices.Camera.Tau.tau2_config as ptc
from devices.Camera.Tau.TauCameraCtrl import Tau
from devices.Camera.utils import connect_ftdi, is_8bit_image_borders_valid, BytesBuffer, \
    REPLY_HEADER_BYTES, parse_incoming_message, make_packet, generate_subsets_indices_in_string
from utils.logger import make_logger, make_logging_handlers

KELVIN2CELSIUS = 273.15
FTDI_PACKET_SIZE = 512 * 8
SYNC_MSG = b'SYNC' + struct.pack(4 * 'B', *[0, 0, 0, 0])


class Tau2Grabber(Tau):
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
        except (RuntimeError, USBError):
            raise RuntimeError('Could not connect to the Tau2 camera.')
        self._lock_parse_command = th.Lock()
        self._event_read = th.Event()
        self._event_read.clear()
        self._event_reply_ready = th.Event()
        self._event_reply_ready.clear()
        self._event_frame_header_in_buffer = th.Event()
        self._event_frame_header_in_buffer.clear()

        self._frame_size = 2 * self.height * self.width + 6 + 4 * self.height  # 6 byte header, 4 bytes pad per row
        self._len_command_in_bytes = 0

        self._buffer = BytesBuffer(size_to_signal=self._frame_size)

        self._thread_read = th.Thread(target=self._th_reader_func, name='th_tau2grabber_reader', daemon=True)
        self._thread_read.start()
        self._log.info('Ready.')

    def __del__(self) -> None:
        if hasattr(self, '_ftdi') and isinstance(self._ftdi, Ftdi):
            self._ftdi.close()
        if hasattr(self, '_event_reply_ready') and isinstance(self._event_reply_ready, th.Event):
            self._event_reply_ready.set()
        if hasattr(self, '_event_frame_header_in_buffer') and isinstance(self._event_frame_header_in_buffer, th.Event):
            self._event_frame_header_in_buffer.set()
        if hasattr(self, '_event_read') and isinstance(self._event_read, th.Event):
            self._event_read.set()
        if hasattr(self, '_log') and isinstance(self._log, logging.Logger):
            try:
                self._log.critical('Exit.')
            except NameError:
                pass

    def _write(self, data: bytes) -> None:
        buffer = b"UART"
        buffer += int(len(data)).to_bytes(1, byteorder='big')  # doesn't matter
        buffer += data
        try:
            self._ftdi.write_data(buffer)
            self._log.debug(f"Send {data}")
        except FtdiError:
            self._log.debug('Write error.')

    def set_params_by_dict(self, yaml_or_dict: (Path, dict)):
        if isinstance(yaml_or_dict, Path):
            params = yaml.safe_load(yaml_or_dict)
        else:
            params = yaml_or_dict.copy()
        self.ace = params.get('ace', 0)
        self.tlinear = params.get('tlinear', 0)
        self.isotherm = params.get('isotherm', 0)
        self.dde = params.get('dde', 0)
        self.gain = params.get('gain', 'high')
        self.agc = params.get('agc', 'manual')
        self.sso = params.get('sso', 0)
        self.contrast = params.get('contrast', 0)
        self.brightness = params.get('brightness', 0)
        self.brightness_bias = params.get('brightness_bias', 0)
        self.cmos_depth = params.get('cmos_depth', 0)  # 14bit pre AGC
        self.fps = params.get('fps', ptc.FPS_CODE_DICT[60])  # 60Hz NTSC
        self.ffc_mode = params.get('ffc_mode', 'external')
        # self.correction_mask = params.get('corr_mask', 0)  # Always OFF!!!

    def _th_reader_func(self) -> None:
        while True:
            self._event_read.wait()
            try:
                data = self._ftdi.read_data(FTDI_PACKET_SIZE)
            except FtdiError:
                return None
            if data is not None and isinstance(self._buffer, BytesBuffer):
                self._buffer += data
            if len(generate_subsets_indices_in_string(self._buffer, b'UART')) == self._len_command_in_bytes:
                self._event_reply_ready.set()
                self._event_read.clear()

    def send_command(self, command: ptc.Code, argument: (bytes, None)) -> (None, bytes):
        data = make_packet(command, argument)
        with self._lock_parse_command:
            self._buffer.clear_buffer()  # ready for the reply
            self._len_command_in_bytes = command.reply_bytes + REPLY_HEADER_BYTES
            self._event_read.set()
            self._write(data)
            self._event_reply_ready.clear()  # counts the number of bytes in the buffer
            self._event_reply_ready.wait(timeout=10)  # blocking until the number of bytes for the reply are reached
            parsed_msg = parse_incoming_message(buffer=self._buffer.buffer, command=command)
            self._event_read.clear()
            if parsed_msg is not None:
                self._log.debug(f"Received {parsed_msg}")
        return parsed_msg

    def grab(self, to_temperature: bool = False):
        with self._lock_parse_command:
            self._buffer.clear_buffer()

            while not self._buffer.sync_teax():
                self._buffer += self._ftdi.read_data(FTDI_PACKET_SIZE)

            while len(self._buffer) < self._frame_size:
                self._buffer += self._ftdi.read_data(min(FTDI_PACKET_SIZE, self._frame_size - len(self._buffer)))

            res = self._buffer[:self._frame_size]
        if not res:
            return None
        magic_word = struct.unpack('h', res[6:8])[0]
        frame_width = struct.unpack('h', res[1:3])[0] - 2
        if magic_word != 0x4000 or frame_width != self.width:
            return None
        raw_image_8bit = np.frombuffer(res[6:], dtype='uint8')
        if len(raw_image_8bit) != (2 * (self.width + 2)) * self.height:
            return None
        raw_image_8bit = raw_image_8bit.reshape((-1, 2 * (self.width + 2)))
        if not is_8bit_image_borders_valid(raw_image_8bit, self.height):
            return None

        raw_image_16bit = 0x3FFF & np.array(raw_image_8bit).view('uint16')[:, 1:-1]
        if to_temperature:
            raw_image_16bit = 0.04 * raw_image_16bit - KELVIN2CELSIUS
        return raw_image_16bit
