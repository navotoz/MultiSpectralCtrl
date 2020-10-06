from utils.logger import make_logger
from vimba import Vimba, MONO_PIXEL_FORMATS
from vimba.error import VimbaTimeout, VimbaFeatureError
from PIL import Image
from devices import CameraAbstract
from devices.AlliedVision.specs import *
from devices.AlliedVision import get_alliedvision_camera_model_name


class AlliedVisionCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        with Vimba.get_instance() as vimba:
            camera_list = vimba.get_all_cameras()
            if not camera_list:
                raise RuntimeError('AlliedVision cameras were not detected. '
                                   'Check if cameras are connected to USB3 via USB3 cable.')
            if model_name:
                func = lambda x: model_name.lower() in get_alliedvision_camera_model_name(x).lower()
                camera_device = list(filter(func, camera_list))[0]
            else:
                camera_device = camera_list[0]
            self._model_name = get_alliedvision_camera_model_name(camera_device)
            self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
            super().__init__(self._model_name, self._log)
            with camera_device as cam:
                cam.AcquisitionMode.set(0) if int(cam.AcquisitionMode.get()) != 0 else None  # single image
                pix_format_max = cam.get_pixel_formats()[-1]
                cam.set_pixel_format(pix_format_max) if cam.get_pixel_format() != pix_format_max else None
            self._camera_device = camera_device
            self.focal_length = CAMERAS_SPECS_DICT[self.model_name].get('focal_length', -1)
            self.f_number = CAMERAS_SPECS_DICT[self.model_name].get('f_number', -1)
            self._log.info(f"Initialized {self.model_name} AlliedVision cameras.")

    @property
    def is_dummy(self):
        return False

    def __call__(self) -> Image.Image:
        with Vimba.get_instance() as _:
            with self._camera_device as cam:
                cam.Gain.set(self.gain) if self.gain is not None and self.gain != cam.Gain.get() else None
                try:
                    cam.ExposureAuto.set(self.exposure_auto)
                except (VimbaFeatureError, AttributeError):
                    pass
                if cam.ExposureTime.get() != self.exposure_time:
                    cam.ExposureTime.set(self.exposure_time) if self.exposure_time is not None else None
                cam.Gamma.set(self.gamma) if self.gamma is not None and self.gamma != cam.Gamma.get() else None
                try:
                    frame = cam.get_frame()
                except VimbaTimeout:
                    self._log.error(f"Camera timed out. Maybe try to reconnect it.")
                    raise TimeoutError(f"Camera timed out. Maybe try to reconnect it.")
                frame.convert_pixel_format(MONO_PIXEL_FORMATS[7])
                frame = frame.as_numpy_ndarray().squeeze()
                return Image.fromarray(frame)
