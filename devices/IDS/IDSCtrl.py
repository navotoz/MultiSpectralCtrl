import sys
import traceback

from devices.IDS.pypyueye.utils import uEyeException
from collections import deque
from utils.constants import MANUAL_EXPOSURE
from utils.logger import make_logger, make_device_logging_handler
from devices import CameraAbstract, get_camera_model_name
from devices import SPECS_DICT
from pyueye import ueye
from devices.IDS.pypyueye import Camera
from numpy import ndarray, ones
from multiprocessing import RLock
from functools import wraps
from server.utils import decorate_all_functions

_lock_ids_ = RLock()


def lock(func):
    @wraps(func)
    def wrapper(*args, **kw):
        with _lock_ids_:
            try:
                return func(*args, **kw)
            except RuntimeError as err:
                raise RuntimeError(err)
            except Exception as err:
                traceback.print_exc(file=sys.stdout)
                print(str(func), err)
    return wrapper


@decorate_all_functions(lock)
class IDSCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        self._camera = Camera()
        self._sensor_info = ueye.SENSORINFO()
        self._camera_info = ueye.CAMINFO()
        self._init_camera()
        ueye.is_GetSensorInfo(self._camera.h_cam, self._sensor_info)
        ueye.is_GetCameraInfo(self._camera.h_cam, self._camera_info)
        self._h, self._w = self._sensor_info.nMaxHeight.value, self._sensor_info.nMaxWidth.value
        self._model_name = get_camera_model_name(self._sensor_info.strSensorName.decode())
        logging_handlers = make_device_logging_handler(f"{self._model_name}", logging_handlers)
        self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
        super().__init__(self._model_name, self._log)
        self.focal_length = SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = SPECS_DICT[self.model_name].get('f_number', -1)
        self._camera.set_gain_auto(0)
        self._log.info(f"Initialized {self.model_name} IDS cameras.")

    def __exit__(self, _type, value, traceback):
        self._log.debug('Exit')
        self._camera.__exit__(_type, value, traceback)

    def __del__(self):
        try:
            self._log.debug('Deleted')
        except AttributeError:
            pass
        self._camera.__exit__(None, None, None)

    def _init_camera(self):
        try:
            self._camera.__enter__()
        except Exception as err:
            raise RuntimeError(f"IDS camera could not be initiated: {err}")

    def _reset(self):
        self._camera.__exit__(None, None, None)
        self._camera = Camera()
        self._init_camera()
        self._log.debug('Reset')

    @property
    def exposure_auto(self) -> str:
        return self._exposure_auto

    @exposure_auto.setter
    def exposure_auto(self, mode: (str, bool)):
        self._set_inner_exposure_auto(mode)
        use_exposure_auto = False if MANUAL_EXPOSURE in self.exposure_auto else True
        self._camera.set_exposure_auto(use_exposure_auto)

    @property
    def aoi(self)->tuple:
        return self._camera.get_aoi()

    @aoi.setter
    def aoi(self, args):
        y, x = args
        x, y = int(x), int(y)
        new_h = self._h - y * 2
        new_w =  self._w - x * 2
        if x >= 16:   # magic number
            self._camera.set_aoi(x, y, new_w, new_h)

    @property
    def is_dummy(self):
        return False

    def _grab(self):
        for _ in range(2):
            for _ in range(5):
                try:
                    return self._camera.capture_image()
                except Exception as err:
                    traceback.print_exc(file=sys.stdout)
                    self._log.warning(f"Failed to capture image due to {err}")
            self._log.error(f"Failed to capture image")
            self._reset()
        return ones((self._h, self._w)).astype('uint8')

    def __call__(self) -> ndarray:
        exposure_time_to_set_milliseconds = self.exposure_time * 1e-3
        use_exposure_auto = False if MANUAL_EXPOSURE in self.exposure_auto else True
        if not use_exposure_auto and self._camera.get_exposure().value != exposure_time_to_set_milliseconds:
            self._camera.set_exposure(exposure_time_to_set_milliseconds)
            self.exposure_time = self._camera.get_exposure().value * 1e3

        self._camera.set_colormode(ueye.IS_CM_MONO8)  # todo: is this the only relevant colormode?

        if use_exposure_auto:
            queue = deque(maxlen=6)
            for _ in range(200):
                self._grab()
                val = self._camera.get_exposure().value
                queue.append(val)
                diff = list(map(lambda x: abs(x - val) < 5e-2, queue))
                if diff and len(diff) == 6 and all(diff):
                    break
        image = self._grab()
        self._log.info(f"Image was taken with #{self.f_number}, focal length {self.focal_length}mm, "
                       f"gain {self.gain}dB, gamma {self.gamma}, "
                       f"exposure {self._camera.get_exposure().value:.2f}milliseconds")
        return image
