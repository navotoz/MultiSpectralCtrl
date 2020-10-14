from utils.logger import make_logger
from PIL import Image
from devices import CameraAbstract, get_camera_model_name
from devices import SPECS_DICT
from server.utils import numpy_to_base64
from pyueye import ueye
from devices.IDS.pypyueye import Camera


class IDSCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        self._camera = Camera()
        self._sensor_info = ueye.SENSORINFO()
        self._camera_info = ueye.CAMINFO()
        with self as cam:
            ueye.is_GetSensorInfo(cam.h_cam, self._sensor_info)
            ueye.is_GetCameraInfo(cam.h_cam, self._camera_info)
        self._model_name = get_camera_model_name(self._sensor_info.strSensorName.decode())
        self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
        super().__init__(self._model_name, self._log)
        self.focal_length = SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = SPECS_DICT[self.model_name].get('f_number', -1)
        self._log.info(f"Initialized {self.model_name} IDS cameras.")

    def __enter__(self):
        try:
            return self._camera.__enter__()
        except:
            raise RuntimeError('IDS camera could not be initiated.')

    def __exit__(self, _type, value, traceback):
        self._camera.__exit__(_type, value, traceback)
        self._camera = Camera()

    @property
    def is_dummy(self):
        return False

    def __call__(self) -> Image.Image:
        with self as cam:
            cam.set_gain_auto(0)
            cam.set_exposure_auto(0 if 'Off' in self.exposure_auto else 1)
            if cam.get_exposure() != self.exposure_time:
                cam.set_exposure(self.exposure_time) if self.exposure_time is not None else None

            cam.set_colormode(ueye.IS_CM_MONO8)  # todo: is this the only relevant colormode?

            image = cam.capture_image()
            return Image.fromarray(image)
