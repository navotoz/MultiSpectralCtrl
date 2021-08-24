import multiprocessing as mp
import threading as th
from itertools import cycle
from time import sleep

import utils.constants as const
from devices import DeviceAbstract
from devices.Camera import CameraAbstract
from devices.Camera.Tau.Tau2Grabber import Tau2Grabber
from utils.logger import make_logging_handlers
from utils.tools import DuplexPipe


class CameraCtrl(DeviceAbstract):
    def _th_cmd_parser(self):
        pass

    _workers_dict = dict()
    _camera: (CameraAbstract, None) = None

    def __init__(self,
                 logging_handlers: (tuple, list),
                 event_stop: mp.Event,
                 image_pipe: DuplexPipe,
                 flag_alive: mp.Event):
        super(CameraCtrl, self).__init__(event_stop, logging_handlers)
        self._values_dict = {}
        self._image_pipe = image_pipe
        self._flags_pipes_list = [self._image_pipe.flag_run]
        self._flag_alive = flag_alive
        self._flag_alive.clear()
        self._lock_camera = th.RLock()
        self._event_get_temperatures = th.Event()
        self._event_get_temperatures.set()

    def _terminate_device_specifics(self):
        try:
            if hasattr(self, '_flag_alive'):
                self._flag_alive.set()
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        try:
            if hasattr(self, '_event_get_temperatures'):
                self._event_get_temperatures.set()
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass
        try:
            self._lock_camera.release()
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

    def _run(self):
        self._workers_dict['connect'] = th.Thread(target=self._th_connect, name='connect', daemon=True)
        self._workers_dict['connect'].start()

        self._workers_dict['getter_t'] = th.Thread(target=self._th_get_temperatures, name='getter_t', daemon=True)
        self._workers_dict['getter_t'].start()

        self._workers_dict['getter_image'] = th.Thread(target=self._th_image_sender, name='getter_image', daemon=True)
        self._workers_dict['getter_image'].start()

    def _getter_temperature(self, t_type: str):
        with self._lock_camera:
            t = self._camera.get_inner_temperature(t_type)
        if t is not None and t != 0.0 and t != -float('inf'):
            try:
                self._values_dict[t_type] = t
            except (BrokenPipeError, RuntimeError):
                pass

    def _th_connect(self):
        handlers = make_logging_handlers(None, True)
        while True:
            with self._lock_camera:
                if not isinstance(self._camera, CameraAbstract):
                    try:
                        camera = Tau2Grabber(logging_handlers=handlers)
                        self._camera = camera
                        self._camera.set_params_by_dict(const.INIT_CAMERA_PARAMETERS)
                        self._getter_temperature(const.T_FPA)
                        self._getter_temperature(const.T_HOUSING)
                        self._flag_alive.set()
                        return
                    except (RuntimeError, BrokenPipeError):
                        pass
            sleep(1)

    def _th_get_temperatures(self) -> None:
        self._flag_alive.wait()
        for t_type in cycle([const.T_FPA, const.T_HOUSING]):
            self._event_get_temperatures.wait(timeout=60 * 10)
            self._getter_temperature(t_type=t_type)
            sleep(const.FREQ_INNER_TEMPERATURE_SECONDS)

    def _th_image_sender(self):
        self._flag_alive.wait()
        while True:
            n_images_to_grab = self._image_pipe.recv()
            if n_images_to_grab is None or n_images_to_grab <= 0:
                self._image_pipe.send(None)
                continue

            images = {}
            for n_image in range(1, n_images_to_grab + 1):
                t_fpa = round(round(self._values_dict[const.T_FPA] * 100), -1)  # precision for the fpa is 0.1C
                t_housing = round(self._values_dict[const.T_HOUSING] * 100)  # precision of the housing is 0.01C
                self._event_get_temperatures.clear()
                with self._lock_camera:
                    images[(t_fpa, t_housing, n_image)] = self._camera.grab() if self._camera is not None else None
                self._event_get_temperatures.set()
            self._image_pipe.send(images)
