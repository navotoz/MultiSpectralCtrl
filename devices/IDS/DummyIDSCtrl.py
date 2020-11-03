import numpy as np
import random
from utils.constants import FAILURE_PROBABILITY_IN_DUMMIES
from utils.logger import make_logger
from devices import CameraAbstract
from devices import SPECS_DICT


class IDSCtrl(CameraAbstract):
    def __init__(self, model_name: (str, None) = None, logging_handlers: (list, tuple) = ()):
        if random.random() < FAILURE_PROBABILITY_IN_DUMMIES:
            raise RuntimeError('Dummy AlliedVisionCtrl simulates failure.')
        if not model_name:
            raise RuntimeError('Cannot initialize a dummy AlliedVision camera without a model name.')
        self._model_name = model_name
        self._log = make_logger(f"{self._model_name}", handlers=logging_handlers)
        super().__init__(self._model_name, self._log)
        self._camera_device = None
        self.focal_length = SPECS_DICT[self.model_name].get('focal_length', -1)
        self.f_number = SPECS_DICT[self.model_name].get('f_number', -1)
        self._log.info(f"Initialized Dummy {self.model_name} IDS cameras.")

    def __del__(self):
        try:
            self._log.debug('Deleted')
        except AttributeError:
            pass
        del self

    @property
    def is_dummy(self):
        return True

    def __call__(self) -> np.ndarray:
        h, w = SPECS_DICT[self.model_name]['h'], SPECS_DICT[self.model_name]['w']
        return np.random.rand(h, w)