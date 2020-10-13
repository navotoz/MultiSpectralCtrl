import numpy as np
from utils.logger import make_logger
from vimba import Vimba, MONO_PIXEL_FORMATS
from vimba.error import VimbaTimeout, VimbaFeatureError
from PIL import Image
from devices import CameraAbstract
from devices import SPECS_DICT, get_camera_model_name
from server.utils import numpy_to_base64

ERR_MSG = 'AlliedVision cameras were not detected. Check if cameras are connected to USB3 via USB3 cable.'


class AlliedVisionCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        self._vimba = Vimba.get_instance()
        with self._vimba:
            camera_list = self._vimba.get_all_cameras()
            func = lambda x: model_name and model_name in get_camera_model_name(x)
            camera_list = list(filter(func, list(camera_list)))
            if not camera_list:
                raise RuntimeError(ERR_MSG)
            self._camera = camera_list[0]
            self._model_name = get_camera_model_name(self._camera)
            self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
            super().__init__(self._model_name, self._log)
            with self._camera:
                self._camera.AcquisitionMode.set(0) if int(
                    self._camera.AcquisitionMode.get()) != 0 else None  # single image
                pix_format_max = self._camera.get_pixel_formats()[-1]
                self._camera.set_pixel_format(
                    pix_format_max) if self._camera.get_pixel_format() != pix_format_max else None
        self.focal_length = SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = SPECS_DICT[self.model_name].get('f_number', -1)
        self._log.info(f"Initialized {self.model_name} AlliedVision cameras.")

    @property
    def is_dummy(self):
        return False

    def __take_image(self) -> np.ndarray:
        try:
            frame = self._camera.get_frame()
        except VimbaTimeout:
            self._log.error(f"Camera timed out. Maybe try to reconnect it.")
            raise TimeoutError(f"Camera timed out. Maybe try to reconnect it.")
        frame.convert_pixel_format(MONO_PIXEL_FORMATS[7])
        return frame.as_numpy_ndarray().squeeze()

    def __call__(self) -> Image.Image:
        with self._vimba:
            with self._camera:
                self._camera.Gain.set(self.gain) if self.gain is not None and self.gain != self._camera.Gain.get() else None
                try:
                    self._camera.ExposureAuto.set(self.exposure_auto)
                except (VimbaFeatureError, AttributeError):
                    pass
                if self._camera.ExposureTime.get() != self.exposure_time:
                    self._camera.ExposureTime.set(self.exposure_time) if self.exposure_time is not None else None
                self._camera.Gamma.set(self.gamma) if self.gamma is not None and self.gamma != self._camera.Gamma.get() else None
                return Image.fromarray(self.__take_image())
