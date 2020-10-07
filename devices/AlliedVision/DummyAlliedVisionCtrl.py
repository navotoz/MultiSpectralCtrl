from PIL import Image
from base64 import b64encode, b64decode
from io import BytesIO

import numpy as np
import random
from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
from utils.logger import make_logger
from devices import CameraAbstract
from devices.AlliedVision.specs import *
from server.utils import numpy_to_base64


class AlliedVisionCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy AlliedVisionCtrl simulates failure.')
        if not model_name:
            raise RuntimeError('Cannot initialize a dummy AlliedVision camera without a model name.')
        self._model_name = model_name
        self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
        super().__init__(self._model_name, self._log)
        self._camera_device = None
        self.focal_length = CAMERAS_SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = CAMERAS_SPECS_DICT[self.model_name].get('f_number', -1)
        self._log.info(f"Initialized Dummy {self.model_name} AlliedVision cameras.")

    def __del__(self):
        pass

    @property
    def is_dummy(self):
        return True

    def stream(self):
        h, w = CAMERAS_SPECS_DICT[self.model_name]['h'], CAMERAS_SPECS_DICT[self.model_name]['w']
        while True:
            if not self.flag_stream:
                yield b''
            else:
                while self.flag_stream:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + numpy_to_base64(np.random.rand(w,h)) + b'\r\n'

    def __call__(self):
        h, w = CAMERAS_SPECS_DICT[self.model_name]['h'], CAMERAS_SPECS_DICT[self.model_name]['w']
        return np.random.rand(h, w)
