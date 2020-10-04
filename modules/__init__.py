def get_alliedvision_grabber(focal_length_mm: (int, float), f_number: (int, float),
                             logging_handlers: (list, tuple), camera_model: str = 'ALVIUM_1800U_1236',
                             use_dummy_filterwheel: bool = False, use_dummy_alliedvision_camera: bool = False):
    if use_dummy_alliedvision_camera:
        from .dummy_AlliedVisionCtrl import DummyAlliedVisionGrabber
        return DummyAlliedVisionGrabber(logging_handlers=logging_handlers)
    else:
        from .AlliedVisionCameraCtrl import AlliedVisionCtrl
        return AlliedVisionCtrl(logging_handlers=logging_handlers)
