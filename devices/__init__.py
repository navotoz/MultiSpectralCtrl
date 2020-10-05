from importlib import import_module
from abc import ABC, abstractmethod, abstractproperty


def initialize_device(element_name: str, handlers: list, use_dummy: bool) -> object:
    use_dummy = 'Dummy' if use_dummy else ''
    if 'filterwheel' in element_name.lower():
        m = import_module(f"devices.{use_dummy}FilterWheel", f"{use_dummy}FilterWheel").FilterWheel
    else:
        raise TypeError(f"{element_name} was not implemented as a module.")
    return m(logging_handlers=handlers)


def get_cameras_module(handlers: list, use_dummy: bool):
    m = import_module(f"devices.CamerasCtrl", f"CamerasCtrl").CamerasCtrl
    return m(logging_handlers=handlers, use_dummy=use_dummy)


class CameraAbstract:
    @abstractmethod
    def __init__(self, logging_handlers: (list, tuple) = ()):
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def is_dummy(self):
        pass

    @property
    @abstractmethod
    def focal_length(self):
        pass

    @property
    @abstractmethod
    def f_number(self):
        pass

    @property
    @abstractmethod
    def gain(self):
        pass

    @property
    @abstractmethod
    def gamma(self):
        pass

    @property
    @abstractmethod
    def exposure_time(self):
        pass

    @property
    @abstractmethod
    def exposure_auto(self) -> str:
        pass


class FilterWheelAbstract:
    __reversed_pos_names_dict = dict()
    __log = None


    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, type_, value, traceback):
        pass

    @property
    @abstractmethod
    def is_dummy(self) -> bool:
        pass

    @property
    @abstractmethod
    def position(self) -> dict:
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def speed(self):
        pass

    @property
    @abstractmethod
    def position_count(self):
        pass

    @property
    @abstractmethod
    def position_names_dict(self):
        pass

    def is_position_name_valid(self, name: str) -> bool:
        """
        Checks if the given name is indeed in the position dict.
        Args:
            name: a string of the name of a filter.
        Returns:
            True if the name is in the position dict or False if the filter name is not in the dictionary.
        """
        return name in self.position_names_dict.values()

    def get_position_from_name(self, name: str) -> int:
        """
        Returns the position of the input on the FilterWheel if valid, else -1.

        Args:
            name a string with a filter name
        Returns:
            The position of the given name on the FilterWheel if a valid, else -1.
        """
        if not self.is_position_name_valid(name):
            self.__log.warning(f"Given position name {name} not in position names dict.")
            return -1
        return self.__reversed_pos_names_dict[name]