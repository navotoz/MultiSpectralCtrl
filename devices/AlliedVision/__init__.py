from importlib import import_module


def init_alliedvision_camera(model_name:(str, None), handlers: (list, tuple), use_dummy: bool):
    use_dummy = 'Dummy' if use_dummy else ''
    m = import_module(f"devices.AlliedVision.{use_dummy}AlliedVisionCameraCtrl", f"{use_dummy}AlliedVisionCtrl").AlliedVisionCtrl
    return m(model_name=model_name, logging_handlers=handlers)
