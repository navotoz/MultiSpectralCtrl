from typing import Dict
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
            self.focal_length = CAMERAS_SPECS_DICT[self.model_name].get('focal_length', -1)
            self.f_number = CAMERAS_SPECS_DICT[self.model_name].get('f_number', -1)
            self._log.info(f"Initialized {self.model_name} AlliedVision cameras.")

    @property
    def is_dummy(self):
        return False

    # @property
    # def image_specs(self):
    #     with Vimba.get_instance() as vimba:
    #         with vimba.get_all_cameras()[0] as cam:
    #             return {**get_camera_features_dict(cam),
    #                     **self.__lens_specs,
    #                     **self.camera_specs}

    def __take_image(self, camera_to_use: CameraAbstract) -> Image.Image:
        """
        Takes an image with the camera. First sets some defaults.

        Returns: Image.Image: the image taken by the camera in uint16.

        Raises: TimeoutError if camera time-out.
        """
        with camera_to_use as cam:
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
                frame.convert_pixel_format(MONO_PIXEL_FORMATS[7])
                frame = frame.as_numpy_ndarray().squeeze()
            except VimbaTimeout:
                self._log.error(f"Camera timed out. Maybe try to reconnect it.")
                raise TimeoutError(f"Camera timed out. Maybe try to reconnect it.")
            return Image.fromarray(frame)

    def __call__(self, camera_model_name: (str, None) = None) -> Dict[str, Image.Image]:
        dict_images, camera = {}, None
        if isinstance(camera_model_name, str):
            camera = self.cameras_dict.get(camera_model_name, None)
        with Vimba.get_instance() as _:
            if camera:
                dict_images[camera_model_name] = self.__take_image(camera)
            else:
                for model, camera in self.cameras_dict.items():
                    dict_images[model] = self.__take_image(camera)
            return dict_images

    def parse_specs_to_tiff(self) -> dict:
        """
        Parse the specs of the camera and download into the TIFF tags format.
        See https://www.awaresystems.be/imaging/tiff/tifftags.html,
        https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
        :return: dict - keys are the TIFF TAGS and values are respective values.
        """
        specs = self.image_specs
        return dict(((272, self.camera_model),
                     (258, f"{specs.get('bit_depth', '12'):}"),
                     (33434, f"{specs.get('exposure_time', 5000.0)}"),
                     (41991, f"{specs.get('gain', 0)}"),
                     (37386, f"{specs.get('focal_length_mm', 12)}"),
                     (33437, f"{specs.get('f_number', 1.4)}"),
                     (37500, f"PixelPitch{specs.get('pixel_size', 0.00345)};"
                             f"SensorHeight{specs.get('sensor_size_h', 14.2)};"
                             f"SensorWidth{specs.get('sensor_size_w', 10.4)};"
                             f"SensorDiagonal{specs.get('sensor_size_diag', 17.6)};")))
