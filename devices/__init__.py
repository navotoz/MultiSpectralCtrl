from numpy import ndarray
from functools import partial
from logging import Logger
from importlib import import_module
from abc import abstractmethod
from devices.AlliedVision import init_alliedvision_camera
from devices.IDS import init_ids_camera
from utils.constants import TIFF_MODEL_NAME, TIFF_GAIN, TIFF_F_NUMBER, TIFF_EXPOSURE_TIME, TIFF_FOCAL_LENGTH, \
    TIFF_NOTES, MANUAL_EXPOSURE, AUTO_EXPOSURE

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
        self._focal_length: float = -1
        self._f_number: float = -1
        self._gamma: float = 1.0
        self._gain: float = 0.0
        self._exposure_time: float = 5000.
        self._exposure_auto: (str, None) = MANUAL_EXPOSURE

    @property
    @abstractmethod
    def is_dummy(self) -> bool:
        pass

    @abstractmethod
    def __call__(self) -> ndarray:
        pass

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def focal_length(self) -> float:
        return self._focal_length

    @focal_length.setter
    def focal_length(self, focal_length_to_set: (float, int)):
        if self.focal_length == focal_length_to_set:
            return
        self._focal_length = float(focal_length_to_set)
        self._log.debug(f"Set focal length to {focal_length_to_set}mm.")

    @property
    def f_number(self) -> float:
        return self._f_number

    @f_number.setter
    def f_number(self, f_number_to_set: (float, int)):
        if self._f_number == f_number_to_set:
            return
        self._f_number = float(f_number_to_set)
        self._log.debug(f"Set f# to {f_number_to_set}.")

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, gain_to_set: (float, int)):
        if self.gain == gain_to_set:
            return
        self._gain = float(gain_to_set)
        self._log.debug(f"Set gain to {gain_to_set}dB.")

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma_to_set: (float, int)):
        if self.gamma == gamma_to_set:
            return
        self._gamma = float(gamma_to_set)
        self._log.debug(f"Set gamma to {gamma_to_set}.")

    @property
    def exposure_time(self) -> float:
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time_to_set: (float, int)):
        if self.exposure_time == exposure_time_to_set:
            return
        self._exposure_time = float(exposure_time_to_set)
        self._log.debug(f"Set exposure time to {exposure_time_to_set} micro seconds.")

    @property
    @abstractmethod
    def exposure_auto(self):
        pass

    @exposure_auto.setter
    @abstractmethod
    def exposure_auto(self, mode: (str, bool)) -> None:
        pass

    def _set_inner_exposure_auto(self, mode: (str, bool)) -> None:
        if not FEATURES_DICT[self.model_name].get('autoexposure', False):
            self._exposure_auto = MANUAL_EXPOSURE
            return
        if isinstance(mode, str):
            self._exposure_auto = mode.capitalize()
        else:
            self._exposure_auto = AUTO_EXPOSURE if mode else MANUAL_EXPOSURE
        self._log.debug(f'Set to {self._exposure_auto} auto exposure mode.')

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
                     (TIFF_NOTES, f"PixelPitch{SPECS_DICT[self.model_name].get('pixel_size', 0.00345)};"
                                  f"SensorHeight{SPECS_DICT[self.model_name].get('sensor_size_h', 14.2)};"
                                  f"SensorWidth{SPECS_DICT[self.model_name].get('sensor_size_w', 10.4)};"
                                  f"SensorDiagonal{SPECS_DICT[self.model_name].get('sensor_size_diag', 17.6)};")))
