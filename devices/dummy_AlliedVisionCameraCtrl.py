import numpy as np
import random
from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
from typing import Dict
from utils.logger import make_logger
from PIL import Image
from devices.camera_specs import CAMERAS_SPECS_DICT


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


def camera_model(camera):
    # if ''
    model = ''.join(map(lambda x: x.lower().capitalize(), model.replace(' ', '-').split('-')))
    return model


class AlliedVisionCtrl:
    camera_specs = _gain = _exposure_time = _gamma = _auto_exposure = None
    cameras_dict = dict()

    def __init__(self, logging_handlers: (list, tuple) = ()):
        self.__log = make_logger(f"DummyAlliedVisionCtrl", handlers=logging_handlers)

        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy FilterWheel simulates failure.')

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
