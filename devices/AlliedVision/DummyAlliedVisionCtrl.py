import random
from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
from typing import Dict
from utils.logger import make_logger
from PIL import Image
from devices import CameraAbstract
from devices.AlliedVision.specs import *


class AlliedVisionCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy AlliedVisionCtrl simulates failure.')
        if not model_name:
            raise RuntimeError('Cannot initialize a dummy AlliedVision camera without a model name.')
        self._model_name = model_name
        self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
        super().__init__(self._model_name, self._log)
        self.focal_length = CAMERAS_SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = CAMERAS_SPECS_DICT[self.model_name].get('f_number', -1)
        self._log.info(f"Initialized Dummy {self.model_name} AlliedVision cameras.")

    @property
    def is_dummy(self):
        return True

    # @property
    # def image_specs(self):
    #     with Vimba.get_instance() as vimba:
    #         with vimba.get_all_cameras()[0] as cam:
    #             return {**get_camera_features_dict(cam),
    #                     **self.__lens_specs,
    #                     **self.camera_specs}

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
