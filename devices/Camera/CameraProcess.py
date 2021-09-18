import multiprocessing as mp
import threading as th
from ctypes import c_ushort, c_byte
from itertools import cycle
from time import sleep

from numpy import copyto, frombuffer, uint16
from usb.core import USBError

import utils.constants as const
from devices import DeviceAbstract
from devices.Camera import CameraAbstract
from devices.Camera.Tau.Tau2Grabber import Tau2Grabber
from utils.logger import make_logging_handlers


class CameraCtrl(DeviceAbstract):
    _camera: (CameraAbstract, None) = None

    def __init__(self, camera_parameters: dict = const.INIT_CAMERA_PARAMETERS):
        super(CameraCtrl, self).__init__()
        self._flag_alive = mp.Event()
        self._flag_alive.clear()
        self._lock_camera = th.RLock()
        self._lock_image = mp.Lock()
        self._event_new_image = mp.Event()
        self._event_new_image.clear()
        self._semaphore_ffc_do = mp.Semaphore(value=0)
        self._semaphore_ffc_finished = mp.Semaphore(value=0)
        self._ffc_result: mp.Value = mp.Value(typecode_or_type=c_byte)
        self._ffc_result.value = 0

        self._image_array = mp.RawArray(c_ushort, const.HEIGHT_IMAGE_TAU2 * const.WIDTH_IMAGE_TAU2)
        self._image_array = frombuffer(self._image_array, dtype=uint16)
        self._image_array = self._image_array.reshape(const.HEIGHT_IMAGE_TAU2, const.WIDTH_IMAGE_TAU2)

        self._fpa: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16
        self._housing: mp.Value = mp.Value(typecode_or_type=c_ushort)  # uint16

        self._camera_params = camera_parameters

    def _terminate_device_specifics(self):
        try:
            self._semaphore_ffc_do.release()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError):
            pass
        try:
            self._semaphore_ffc_finished.release()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError):
            pass
        try:
            self._event_new_image.set()
        except (ValueError, TypeError, AttributeError, RuntimeError, NameError):
            pass

    def _run(self):
        self._workers_dict['connect_cam'] = th.Thread(target=self._th_connect, name='connect_cam', daemon=True)
        self._workers_dict['connect_cam'].start()

        self._workers_dict['getter_t'] = th.Thread(target=self._th_getter_temperature, name='getter_t', daemon=True)
        self._workers_dict['getter_t'].start()

        self._workers_dict['getter_image'] = th.Thread(target=self._th_getter_image, name='getter_image', daemon=False)
        self._workers_dict['getter_image'].start()

        self._workers_dict['th_ffc'] = th.Thread(target=self._th_ffc_func, name='th_ffc', daemon=True)
        self._workers_dict['th_ffc'].start()

    def _th_connect(self):
        handlers = make_logging_handlers(None, True)
        while True:
            sleep(1)
            with self._lock_camera:
                if not isinstance(self._camera, CameraAbstract):
                    try:
                        self._camera = Tau2Grabber(logging_handlers=handlers)
                        self._camera.set_params_by_dict(self._camera_params)
                        self._getter_temperature(const.T_FPA)
                        self._getter_temperature(const.T_HOUSING)
                        self._flag_alive.set()
                        return
                    except (RuntimeError, BrokenPipeError, USBError):
                        pass

    def _th_ffc_func(self):
        self._flag_alive.wait()
        while self._flag_run:
            self._ffc_result.value = self._camera.ffc()
            self._semaphore_ffc_finished.release()
            self._semaphore_ffc_do.acquire()

    def _getter_temperature(self, t_type: str):  # this function exists for the th_connect function, otherwise redundant
        with self._lock_camera:
            t = self._camera.get_inner_temperature(t_type) if self._camera is not None else None
        if t is not None and t != 0.0 and t != -float('inf'):
            try:
                t = round(t * 100)
                if t_type == const.T_FPA:
                    self._fpa.value = round(t, -1)  # precision for the fpa is 0.1C
                elif t_type == const.T_HOUSING:
                    self._housing.value = t  # precision of the housing is 0.01C
            except (BrokenPipeError, RuntimeError):
                pass

    def _th_getter_temperature(self) -> None:
        self._flag_alive.wait()
        for t_type in cycle([const.T_FPA, const.T_HOUSING]):
            self._getter_temperature(t_type=t_type)
            sleep(const.FREQ_INNER_TEMPERATURE_SECONDS)

    def _th_getter_image(self):
        while not self._flag_alive.wait(timeout=3) and self._flag_alive:
            pass  # this thread is not a Daemon, so live-wait of some sort is performed.
        while self._flag_run:
            with self._lock_camera:
                image = self._camera.grab() if self._camera is not None else None
            if image is not None:
                with self._lock_image:
                    copyto(self._image_array, image)
                    self._event_new_image.set()

    @property
    def image(self):
        self._event_new_image.wait()
        with self._lock_image:
            self._event_new_image.clear()
            return self._image_array.copy()

    def ffc(self) -> bool:
        self._semaphore_ffc_do.release()
        self._semaphore_ffc_finished.acquire()
        return bool(self._ffc_result.value)

    @property
    def fpa(self):
        return self._fpa.value

    @property
    def housing(self):
        return self._housing.value

    @property
    def is_camera_alive(self):
        return self._flag_alive.is_set()
