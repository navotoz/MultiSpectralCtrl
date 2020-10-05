import random
from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
from typing import Dict
from utils.logger import make_logger
from PIL import Image
from devices import CameraAbstract
from devices.AlliedVision.alliedvision_specs import *


def get_camera_features_dict(cam):
    features = list(filter(lambda feat: feat.is_readable(), cam.get_all_features()))
    features = dict(map(lambda feat: (feat.get_name(), feat.get()), features))
    ret_dict = dict(
        exposure_time=features['ExposureTime'],
        gain=features['Gain'],
        gamma=features['Gamma'])
    if features['ContrastEnable']:
        ret_dict['contrast_bright_limit'] = features['ContrastBrightLimit']
        ret_dict['contrast_dark_limit'] = features['ContrastDarkLimit']
    return ret_dict


class AlliedVisionCtrl(CameraAbstract):
    __gain = 0.0
    __gamma = 1.0
    __exposure_auto = 'Off'
    __exposure_time = 5000.
    __model_name = None
    __focal_length = __f_number = None

    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        super().__init__(logging_handlers)
        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy AlliedVisionCtrl simulates failure.')
        if not model_name:
            raise RuntimeError('Cannot initilize a dummy AlliedVision camera without a model name.')
        self.__model_name = model_name
        self.__log = make_logger(f"{self.model_name}", handlers=logging_handlers)
        self.focal_length = CAMERAS_SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = CAMERAS_SPECS_DICT[self.model_name].get('f_number', -1)
        self.__log.info(f"Initialized Dummy {self.model_name} AlliedVision cameras.")


    @property
    def model_name(self):
        return self.__model_name

    @property
    def is_dummy(self):
        return True

    @property
    def focal_length(self):
        return self.__focal_length

    @focal_length.setter
    def focal_length(self, focal_length_to_set):
        if self.focal_length == focal_length_to_set:
            return
        self.__log.debug(f"Set focal length to {focal_length_to_set}mm.")

    @property
    def f_number(self):
        return self.__f_number

    @f_number.setter
    def f_number(self, f_number_to_set):
        if self.focal_length == f_number_to_set:
            return
        self.__log.debug(f"Set f# to {f_number_to_set}mm.")

    # @property
    # def image_specs(self):
    #     with Vimba.get_instance() as vimba:
    #         with vimba.get_all_cameras()[0] as cam:
    #             return {**get_camera_features_dict(cam),
    #                     **self.__lens_specs,
    #                     **self.camera_specs}

    @property
    def gain(self):
        return self.__gain

    @gain.setter
    def gain(self, gain_to_set):
        if self.gain == gain_to_set:
            return
        self.__gain = gain_to_set
        self.__log.debug(f"Set gain to {gain_to_set}dB.")

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma_to_set):
        if self.gamma == gamma_to_set:
            return
        self.__gamma = gamma_to_set
        self.__log.debug(f"Set gamma to {gamma_to_set}.")

    @property
    def exposure_time(self):
        return self.__exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time_to_set):
        if self.exposure_time == exposure_time_to_set:
            return
        self.__exposure_time = exposure_time_to_set
        self.__log.debug(f"Set exposure time to {exposure_time_to_set} micro seconds.")

    @property
    def exposure_auto(self):
        return self.__exposure_auto

    @exposure_auto.setter
    def exposure_auto(self, mode: (str, bool)):
        if not CAMERAS_FEATURES_DICT[self.model_name].get('autoexposure', True):
            self.__exposure_auto = None
            return
        if isinstance(mode, str):
            self.__exposure_auto = mode.capitalize()
        else:
            self.__exposure_auto = 'Once' if mode else 'Off'
        self.__log.debug(f'Set to {self.__exposure_auto} auto exposure mode.')

    def __take_image(self, camera_to_use) -> Image.Image:

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
