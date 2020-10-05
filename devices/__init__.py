from functools import partial
from logging import Logger
from importlib import import_module
from abc import abstractmethod
from devices.AlliedVision.specs import *
from devices.AlliedVision import init_alliedvision_camera

valid_cameras_names_list = [*ALLIEDVISION_VALID_MODEL_NAMES]


def initialize_device(element_name: str, handlers: list, use_dummy: bool) -> object:
    if 'filterwheel' in element_name.lower():
        use_dummy = 'Dummy' if use_dummy else ''
        m = import_module(f"devices.FilterWheel.{use_dummy}FilterWheel", f"{use_dummy}FilterWheel").FilterWheel
    else:
        if element_name in ALLIEDVISION_VALID_MODEL_NAMES:
            m = partial(init_alliedvision_camera, model_name=element_name, use_dummy=use_dummy)
        # todo: add other cameras...
        else:
            raise TypeError(f"{element_name} was not implemented as a module.")
    return m(logging_handlers=handlers)


class CameraAbstract:
    def __init__(self, model_name: str, logger: Logger):
        self._model_name = model_name
        self._log = logger
        self.__focal_length = None
        self.__f_number = None
        self.__gamma = 1.0
        self.__gain = 0.0
        self.__exposure_time = 5000.
        self.__exposure_auto = 'Off'

    @property
    @abstractmethod
    def is_dummy(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @property
    def model_name(self):
        return self._model_name

    @property
    def focal_length(self):
        return self.__focal_length

    @focal_length.setter
    def focal_length(self, focal_length_to_set):
        if self.focal_length == focal_length_to_set:
            return
        self._log.debug(f"Set focal length to {focal_length_to_set}mm.")

    @property
    def f_number(self):
        return self.__f_number

    @f_number.setter
    def f_number(self, f_number_to_set):
        if self.focal_length == f_number_to_set:
            return
        self._log.debug(f"Set f# to {f_number_to_set}.")

    @property
    def gain(self):
        return self.__gain

    @gain.setter
    def gain(self, gain_to_set):
        if self.gain == gain_to_set:
            return
        self.__gain = gain_to_set
        self._log.debug(f"Set gain to {gain_to_set}dB.")

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma_to_set):
        if self.gamma == gamma_to_set:
            return
        self.__gamma = gamma_to_set
        self._log.debug(f"Set gamma to {gamma_to_set}.")

    @property
    def exposure_time(self):
        return self.__exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time_to_set):
        if self.exposure_time == exposure_time_to_set:
            return
        self.__exposure_time = exposure_time_to_set
        self._log.debug(f"Set exposure time to {exposure_time_to_set} micro seconds.")

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
        self._log.debug(f'Set to {self.__exposure_auto} auto exposure mode.')

