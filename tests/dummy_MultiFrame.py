import random
from FilterWheel import FilterWheel
from utils.logger import make_logger, INFO
import numpy as np
from itertools import compress
from time import sleep
from datetime import datetime
from PIL import Image
from utils.camera_specs import CAMERAS_SPECS_DICT
from utils.constants import FILTER_WHEEL_SETTLING_TIME, FAILURE_PROBABILITY


class MultiFrameGrabber:
    __filters_sequence = dict(positions=1)
    __camera_specs = __camera_model = __gain = __exposure_time = __gamma = __auto_exposure = None

    def __init__(self, focal_length_mm: (int, float), f_number: (int, float), logging_handlers: (list, tuple),
                 camera_model: str = 'ALVIUM_1800U_1236', dummy: bool = False):
        self.__log = make_logger('DummyMultiFrameGrabber', handlers=logging_handlers)

        if random.random() < FAILURE_PROBABILITY:
            raise RuntimeError('Dummy MultiFrameGrabber simulated a failure.')

        if not dummy:
            self.__filter_wheel = FilterWheel(logger=make_logger('FilterWheel', logging_handlers, level=INFO))
        else:
            self.__log.warning('Using dummy FilterWheel.')
            from tests.dummy_FilterWheel import DummyFilterWheel
            self.__filter_wheel = DummyFilterWheel(logger=make_logger('DummyFilterWheel', logging_handlers, level=INFO))
        self.camera_model = camera_model
        self.__lens_specs = dict(focal_length_mm=float(focal_length_mm), f_number=float(f_number), units="mm")
        if self.__lens_specs['units'] != self.camera_specs['units']:
            msg = f"The units in the camera spec ({self.__camera_specs['units']}) " \
                  f"and the lens spec ({self.__lens_specs['units']}) mismatch."
            self.__log.error(msg)
            raise ValueError(msg)
        self.__lens_specs.pop('units')
        self.__camera_specs.pop('units')

    @property
    def camera_model(self) -> str:
        return self.__camera_model

    @camera_model.setter
    def camera_model(self, camera_model: str):
        if self.camera_model == camera_model:
            return
        try:
            self.__camera_specs = CAMERAS_SPECS_DICT[camera_model].copy()
            self.__camera_model = camera_model
            self.__log.debug(f"Set camera model to {camera_model}.")
        except KeyError:
            msg = f"Model {camera_model} is invalid. Available models: {list(CAMERAS_SPECS_DICT.keys())}"
            self.__log.error(msg)
            raise ValueError(msg)

    @property
    def camera_specs(self) -> dict:
        return self.__camera_specs

    @property
    def focal_length(self):
        return self.__lens_specs.setdefault('focal_length_mm', 0)

    @focal_length.setter
    def focal_length(self, focal_length_mm):
        if self.focal_length == focal_length_mm:
            return
        self.__lens_specs['focal_length_mm'] = focal_length_mm
        self.__log.debug(f"Set focal length to {focal_length_mm}mm.")

    @property
    def f_number(self):
        return self.__lens_specs.setdefault('f_number', 0)

    @f_number.setter
    def f_number(self, f_number):
        if self.f_number == f_number:
            return
        self.__lens_specs['f_number'] = f_number
        self.__log.debug(f"Set f_number to {f_number}.")

    @property
    def image_specs(self):
        return {'exposure_time': 6000.0, 'gain': 0.0, 'gamma': 1.0,
                **self.__lens_specs,
                **self.__camera_specs}

    @property
    def gain(self):
        return self.__gain

    @gain.setter
    def gain(self, gain):
        if self.gain == gain:
            return
        self.__gain = gain
        self.__log.debug(f"Set gain to {gain}dB.")

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma):
        if self.gamma == gamma:
            return
        self.__gamma = gamma
        self.__log.debug(f"Set gamma to {gamma}.")

    @property
    def exposure_time(self):
        return self.__exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time):
        if self.exposure_time == exposure_time:
            return
        self.__exposure_time = exposure_time
        self.__log.debug(f"Set exposure time to {exposure_time} micro seconds.")

    @property
    def auto_exposure(self):
        return self.__auto_exposure

    @auto_exposure.setter
    def auto_exposure(self, mode: bool):
        self.__auto_exposure = mode
        self.__log.debug(f'Set to {"auto" if mode else "manual"} exposure.')

    @property
    def filter_names_dict(self):
        return self.__filter_wheel.position_names_dict

    @filter_names_dict.setter
    def filter_names_dict(self, names_dict: dict):
        if self.__filter_wheel.position_names_dict == names_dict:
            return
        self.__filter_wheel.position_names_dict = names_dict
        self.filters_sequence = self.filters_sequence['positions']
        self.__log.debug(f'Changed filters names to {list(self.filter_names_dict.values())}.')

    @property
    def filters_sequence(self) -> dict:
        return self.__filters_sequence

    @filters_sequence.setter
    def filters_sequence(self, seq: (tuple, list, int, str)):
        if isinstance(seq, str) or isinstance(seq, int):
            seq = [seq]
        if all(map(lambda x: isinstance(x, int) and self.__filter_wheel.is_position_in_limits(x), seq)):
            positions_list = seq.copy() if isinstance(seq, list) else seq  # the positions themselves are given
        else:  # a list of names is given
            cmp_seq_to_filter_names = list(map(lambda name: self.__filter_wheel.is_position_name_valid(str(name)), seq))
            if not all(cmp_seq_to_filter_names):
                msg = f"These names does not appear in the filter list: " \
                      f"{list(compress(seq, ~np.array(cmp_seq_to_filter_names)))}."
                self.__log.error(msg)
                raise ValueError(msg)
            positions_list = list(map(lambda name: self.__filter_wheel.get_position_from_name(name), seq))
        positions_list.sort()
        self.__filters_sequence = dict(filters=list(map(lambda key: self.filter_names_dict[key], positions_list)),
                                       positions=positions_list)
        msg = f"The sequence is set to: " \
              f"{list(zip(self.filters_sequence['positions'], self.filters_sequence['filters']))}"
        self.__log.debug(msg)

    def take_image(self) -> Image.Image:
        """
        Takes an image with the camera. First sets some defaults.

        Returns: Image.Image: the image taken by the camera in uint16.

        Raises: TimeoutError if camera time-out.
        """
        frame = np.random.rand(3008, 4112)  # PUT REAL H x W in the random creation
        return Image.fromarray(frame)

    def __call__(self) -> tuple:
        """
        Takes a multiframe image.
        Goes through the filter sequence in self.filters_sequence and takes an image in each position.
        Loges the finishing time and creates a filename containing the timestamp and the filter names.

        :returns: A tuple containing:
            - dict: keys as filter names and values are Image.Image for each filter.
            - dict: TIFF TAGS.
            - str: filename with timestamp and filter names.
        """
        multi_frame_images_dict = {}
        self.__log.debug('Beginning sequence.')
        for (position, name) in zip(self.filters_sequence['positions'], self.filters_sequence['filters']):
            self.__filter_wheel.position = position
            sleep(FILTER_WHEEL_SETTLING_TIME)
            try:
                multi_frame_images_dict[name] = self.take_image()  # take a picture
            except TimeoutError:  # give it another try
                multi_frame_images_dict[name] = self.take_image()  # take a picture
            self.__log.debug(f"Taken filter {name} image.")
        self.__log.debug('Finished sequence.')
        timestamp = datetime.now().strftime('d20%y%m%d_h%Hm%Ms%S')
        f_name = f"{timestamp}_{len(multi_frame_images_dict)}Filters"
        for _filter in multi_frame_images_dict.keys():
            f_name += f"_{_filter}"
        self.__log.info(f'Taken image {f_name}.')
        parsed_images_spec_dict = self.parse_specs_to_tiff()
        return multi_frame_images_dict, parsed_images_spec_dict, f_name

    def parse_specs_to_tiff(self) -> dict:
        """
        Parse the specs of the camera and images into the TIFF tags format.
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
