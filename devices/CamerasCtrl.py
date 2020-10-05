from typing import Dict
from utils.logger import make_logger
from vimba import Vimba, Camera, MONO_PIXEL_FORMATS
from vimba.error import VimbaTimeout, VimbaFeatureError
from PIL import Image
from devices.AlliedVision import init_alliedvision_camera
from devices.AlliedVision.alliedvision_specs import ALLIEDVISION_VALID_MODEL_NAMES


valid_model_names_list = [*ALLIEDVISION_VALID_MODEL_NAMES]

class CamerasCtrl:
    camera_specs = _gain = _exposure_time = _gamma = _auto_exposure = None
    cameras_dict = dict()

    def __init__(self, logging_handlers: (list, tuple) = ()):
        self.__log = make_logger(f"CamerasCtrl", handlers=logging_handlers)

    @property
    def is_dummy(self):
        # todo: add other cameras to the check
        return self._alliedvision_cameras.is_dummy

    @property
    def cameras_models_names(self):
        # todo: add other cameras to the check
        return [*self._alliedvision_cameras.models_names]

    # @property
    # def focal_length(self):
    #     return self.__lens_specs.setdefault('focal_length_mm', 0)
    #
    # @focal_length.setter
    # def focal_length(self, focal_length_mm):
    #     if self.focal_length == focal_length_mm:
    #         return
    #     self.__lens_specs['focal_length_mm'] = focal_length_mm
    #     self.__log.debug(f"Set focal length to {focal_length_mm}mm.")
    #
    # @property
    # def f_number(self):
    #     return self.__lens_specs.setdefault('f_number', 0)
    #
    # @f_number.setter
    # def f_number(self, f_number):
    #     if self.f_number == f_number:
    #         return
    #     self.__lens_specs['f_number'] = f_number
    #     self.__log.debug(f"Set f_number to {f_number}.")
    #
    # @property
    # def image_specs(self):
    #     with Vimba.get_instance() as vimba:
    #         with vimba.get_all_cameras()[0] as cam:
    #             return {**get_camera_features_dict(cam),
    #                     **self.__lens_specs,
    #                     **self.camera_specs}
    #
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        if self.gain == gain:
            return
        self._gain = gain
        self.__log.debug(f"Set gain to {gain}dB.")

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if self.gamma == gamma:
            return
        self._gamma = gamma
        self.__log.debug(f"Set gamma to {gamma}.")

    @property
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time):
        if self.exposure_time == exposure_time:
            return
        self._exposure_time = exposure_time
        self.__log.debug(f"Set exposure time to {exposure_time} micro seconds.")

    @property
    def auto_exposure(self):
        return self._auto_exposure

    @auto_exposure.setter
    def auto_exposure(self, mode: (str, bool)):
        if isinstance(mode, str):
            self._auto_exposure = mode.capitalize()
        else:
            self._auto_exposure = 'Once' if mode else 'Off'
        self.__log.debug(f'Set to {self._auto_exposure} auto exposure.')

    def __take_image(self, camera_to_use: Camera) -> Image.Image:
        """
        Takes an image with the camera. First sets some defaults.

        Returns: Image.Image: the image taken by the camera in uint16.

        Raises: TimeoutError if camera time-out.
        """
        with camera_to_use as cam:
            cam.Gain.set(self.gain) if self.gain is not None and self.gain != cam.Gain.get() else None
            try:
                cam.ExposureAuto.set(self.auto_exposure)
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
                self.__log.error(f"Camera timed out. Maybe try to reconnect it.")
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
