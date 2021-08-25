import binascii
import re
import struct
import threading as th
from typing import List

import numpy
import usb
from pyftdi.ftdi import Ftdi

from devices.Camera import _make_device_from_vid_pid
from devices.Camera.Tau.Tau2Grabber import BORDER_VALUE
from devices.Camera.Tau.tau2_config import Code
from utils.tools import SyncFlag
import numpy as np

BUFFER_SIZE = int(2 ** 24)  # 16 MBytes
LEN_TEAX = 4
UART_PREAMBLE_LENGTH = 6
HEADER_SIZE_IN_BYTES = 10


class BytesBuffer:
    def __init__(self, flag_run: SyncFlag, size_to_signal: int = 0) -> None:
        self._buffer: bytes = b''
        self._lock = th.Lock()
        self._event_buffer_bigger_than = th.Event()
        self._event_buffer_bigger_than.clear()
        self._size_to_signal = size_to_signal
        self._flag_run = flag_run

    def wait_for_size(self):
        while not self._event_buffer_bigger_than.wait(timeout=1) and self._flag_run:
            pass

    def __del__(self) -> None:
        if hasattr(self, '_event_buffer_bigger_than') and isinstance(self._event_buffer_bigger_than, th.Event):
            self._event_buffer_bigger_than.set()

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer = b''
            self._event_buffer_bigger_than.clear()

    def sync_teax(self) -> None:
        with self._lock:
            idx_sync = self._buffer.rfind(b'TEAX')
            if idx_sync != -1:
                self._buffer = self._buffer[idx_sync + LEN_TEAX:]
                if len(self._buffer) >= self._size_to_signal:
                    self._event_buffer_bigger_than.set()
                else:
                    self._event_buffer_bigger_than.clear()
                return

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __add__(self, other: bytes):
        with self._lock:
            self._buffer += other
            if len(self._buffer) > BUFFER_SIZE:
                self._buffer = self._buffer[-BUFFER_SIZE:]
            if len(self._buffer) >= self._size_to_signal:
                self._event_buffer_bigger_than.set()
            else:
                self._event_buffer_bigger_than.clear()
            return self._buffer

    def __iadd__(self, other: bytes):
        with self._lock:
            self._buffer += other
            if len(self._buffer) > BUFFER_SIZE:
                self._buffer = self._buffer[-BUFFER_SIZE:]
            if len(self._buffer) >= self._size_to_signal:
                self._event_buffer_bigger_than.set()
            else:
                self._event_buffer_bigger_than.clear()
            return self

    def __getitem__(self, item: slice) -> bytes:
        with self._lock:
            if isinstance(item, slice):
                return self._buffer[item]

    def __call__(self) -> bytes:
        with self._lock:
            return self._buffer

    @property
    def buffer(self) -> bytes:
        with self._lock:
            return self._buffer


def generate_subsets_indices_in_string(input_string: (BytesWarning, bytes, bytes, map, filter),
                                       subset: (bytes, str)) -> list:
    reg = re.compile(subset)
    if isinstance(input_string, BytesBuffer):
        return [i.start() for i in reg.finditer(input_string())]
    return [i.start() for i in reg.finditer(input_string)]


def generate_overlapping_list_chunks(generator: (map, filter), n: int):
    lst = list(generator)
    subset_generator = map(lambda idx: lst[idx:idx + n], range(len(lst)))
    return filter(lambda sub: len(sub) == n, subset_generator)


def get_crc(data) -> List[int]:
    crc = struct.pack(len(data) * 'B', *data)
    crc = binascii.crc_hqx(crc, 0)
    crc = [((crc & 0xFF00) >> 8).to_bytes(1, 'big'), (crc & 0x00FF).to_bytes(1, 'big')]
    return list(map(lambda x: int.from_bytes(x, 'big'), crc))


def connect_ftdi(vid, pid) -> Ftdi:
    device = _make_device_from_vid_pid(vid, pid)

    usb.util.claim_interface(device, 0)
    usb.util.claim_interface(device, 1)

    ftdi = Ftdi()
    ftdi.open_from_device(device)

    ftdi.set_bitmode(0xFF, Ftdi.BitMode.RESET)
    ftdi.set_bitmode(0xFF, Ftdi.BitMode.SYNCFF)
    return ftdi


def _is_8bit_image_borders_valid(raw_image_8bit: np.ndarray, height: int) -> bool:
    if raw_image_8bit is None:
        return False
    try:
        if np.nonzero(raw_image_8bit[:, 0] != 0)[0]:
            return False
    except ValueError:
        return False
    valid_idx = np.nonzero(raw_image_8bit[:, -1] != BORDER_VALUE)
    if len(valid_idx) != 1:
        return False
    valid_idx = int(valid_idx[0])
    if valid_idx != height - 1:  # the different value should be in the bottom of the border
        return False
    return True


def parse_incoming_message(buffer: bytes, command: Code) -> (List, None):
    len_in_bytes = command.reply_bytes + HEADER_SIZE_IN_BYTES
    argument_length = len_in_bytes * UART_PREAMBLE_LENGTH

    idx_list = generate_subsets_indices_in_string(buffer, b'UART')
    if not idx_list:
        return None
    try:
        data = map(lambda idx: buffer[idx:idx + argument_length][5::6], idx_list)
        data = map(lambda d: d[0], data)
        data = generate_overlapping_list_chunks(data, len_in_bytes)
        data = filter(lambda res: len(res) >= len_in_bytes, data)  # length of message at least as expected
        data = filter(lambda res: res[0] == 110, data)  # header is 0x6E (110)
        data = list(filter(lambda res: res[3] == command.code, data))
    except IndexError:
        data = None
    if not data:
        return None
    data = data[-1]
    crc_1 = get_crc(data[:6])
    crc_2 = get_crc(data[8:8 + command.reply_bytes])
    if not crc_1 == data[6:8] or not crc_2 == data[-2:]:
        return None
    ret_value = data[8:8 + command.reply_bytes]
    ret_value = struct.pack('<' + len(ret_value) * 'B', *ret_value)
    return ret_value
