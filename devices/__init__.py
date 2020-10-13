from functools import partial
from logging import Logger
from importlib import import_module
from abc import abstractmethod
from devices.AlliedVision import init_alliedvision_camera
from devices.IDS import init_ids_camera
from PIL.Image import Image

ALLIEDVISION_VALID_MODEL_NAMES = import_module(f"devices.AlliedVision", f"AlliedVision").get_specs_dict().keys()
IDS_VALID_MODEL_NAMES = import_module(f"devices.IDS", f"IDS").get_specs_dict().keys()
FEATURES_DICT = {**import_module(f"devices.AlliedVision", f"AlliedVision").get_features_dict(),
                 **import_module(f"devices.IDS", f"IDS").get_features_dict()}
SPECS_DICT = {**import_module(f"devices.AlliedVision", f"AlliedVision").get_specs_dict(),
              **import_module(f"devices.IDS", f"IDS").get_specs_dict()}
valid_cameras_names_list = [*ALLIEDVISION_VALID_MODEL_NAMES,
                            *IDS_VALID_MODEL_NAMES]


# todo: add other cameras to the list here...


def initialize_device(element_name: str, handlers: list, use_dummy: bool) -> object:
    if 'filterwheel' in element_name.lower():
        use_dummy = 'Dummy' if use_dummy else ''
        m = import_module(f"devices.FilterWheel.{use_dummy}FilterWheel", f"{use_dummy}FilterWheel").FilterWheel
    else:
        if element_name in ALLIEDVISION_VALID_MODEL_NAMES:
            m = partial(init_alliedvision_camera, model_name=element_name, use_dummy=use_dummy)
        elif element_name in IDS_VALID_MODEL_NAMES:
            m = partial(init_ids_camera, model_name=element_name, use_dummy=use_dummy)
        # todo: add other cameras...
        else:
            raise TypeError(f"{element_name} was not implemented as a module.")
    return m(logging_handlers=handlers)


def get_camera_model_name(camera):
    if not isinstance(camera, str):
        model = camera.get_model()
    else:
        model = camera
    model = ''.join(map(lambda x: x.lower().capitalize(), model.replace(' ', '-').split('-')))
    return model


class CameraAbstract:
    def __init__(self, model_name: str, logger: Logger):
        self._model_name: str = model_name
        self._log: Logger = logger
        self.__focal_length: float = -1
        self.__f_number: float = -1
        self.__gamma: float = 1.0
        self.__gain: float = 0.0
        self.__exposure_time: float = 5000.
        self.__exposure_auto: str = 'Off'

    @property
    @abstractmethod
    def is_dummy(self) -> bool:
        pass

    @abstractmethod
    def __call__(self) -> Image:
        pass

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def focal_length(self) -> float:
        return self.__focal_length

    @focal_length.setter
    def focal_length(self, focal_length_to_set: (float, int)):
        if self.focal_length == focal_length_to_set:
            return
        self.__focal_length = float(focal_length_to_set)
        self._log.debug(f"Set focal length to {focal_length_to_set}mm.")

    @property
    def f_number(self) -> float:
        return self.__f_number

    @f_number.setter
    def f_number(self, f_number_to_set: (float, int)):
        if self.focal_length == f_number_to_set:
            return
        self.__f_number = float(f_number_to_set)
        self._log.debug(f"Set f# to {f_number_to_set}.")

    @property
    def gain(self) -> float:
        return self.__gain

    @gain.setter
    def gain(self, gain_to_set: (float, int)):
        if self.gain == gain_to_set:
            return
        self.__gain = float(gain_to_set)
        self._log.debug(f"Set gain to {gain_to_set}dB.")

    @property
    def gamma(self) -> float:
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma_to_set: (float, int)):
        if self.gamma == gamma_to_set:
            return
        self.__gamma = float(gamma_to_set)
        self._log.debug(f"Set gamma to {gamma_to_set}.")

    @property
    def exposure_time(self) -> float:
        return self.__exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time_to_set: (float, int)):
        if self.exposure_time == exposure_time_to_set:
            return
        self.__exposure_time = float(exposure_time_to_set)
        self._log.debug(f"Set exposure time to {exposure_time_to_set} micro seconds.")

    @property
    def exposure_auto(self) -> str:
        return self.__exposure_auto

    @exposure_auto.setter
    def exposure_auto(self, mode: (str, bool)):
        if not FEATURES_DICT[self.model_name].get('autoexposure', True):
            self.__exposure_auto = None
            return
        if isinstance(mode, str):
            self.__exposure_auto = mode.capitalize()
        else:
            self.__exposure_auto = 'Once' if mode else 'Off'
        self._log.debug(f'Set to {self.__exposure_auto} auto exposure mode.')

    def parse_specs_to_tiff(self) -> dict:
        """
        Parse the specs of the camera and download into the TIFF tags format.
        See https://www.awaresystems.be/imaging/tiff/tifftags.html,
        https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
        :return: dict - keys are the TIFF TAGS and values are respective values.
        """
        return dict(((TIFF_MODEL_NAME, f"{self.model_name}"),
                     (258, f"{SPECS_DICT[self.model_name].get('bit_depth', '12')}"),
                     (TIFF_EXPOSURE_TIME, f"{self.exposure_time}"),
                     (TIFF_GAIN, f"{self.gain}"),
                     (TIFF_FOCAL_LENGTH, f"{self.focal_length}"),
                     (TIFF_F_NUMBER, f"{self.f_number}"),
                     (37500, f"PixelPitch{SPECS_DICT[self.model_name].get('pixel_size', 0.00345)};"
                             f"SensorHeight{SPECS_DICT[self.model_name].get('sensor_size_h', 14.2)};"
                             f"SensorWidth{SPECS_DICT[self.model_name].get('sensor_size_w', 10.4)};"
                             f"SensorDiagonal{SPECS_DICT[self.model_name].get('sensor_size_diag', 17.6)};")))


TIFF_MODEL_NAME = 272
TIFF_EXPOSURE_TIME = 33434
TIFF_GAIN = 41991
TIFF_F_NUMBER = 33437
TIFF_FOCAL_LENGTH = 37386
