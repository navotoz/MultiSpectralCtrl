from importlib import import_module


def init_alliedvision_camera(model_name:(str, None), handlers: (list, tuple), use_dummy: bool):
    use_dummy = 'Dummy' if use_dummy else ''
    path = "devices.AlliedVision"
    m = import_module(f"{path}.{use_dummy}AlliedVisionCtrl", f"{use_dummy}AlliedVisionCtrl").AlliedVisionCtrl
    return m(model_name=model_name, logging_handlers=handlers)


def get_camera_features_dict(cam):
    features = list(filter(lambda feat: feat.is_readable(), cam.get_all_features()))
    features = dict(map(lambda feat: (feat.get_name(), feat.get()), features))
    ret_dict = dict(
        exposure_time=features['ExposureTime'],
        gain=features['Gain'],
        gamma=features['Gamma'])
    if features['ContrastEnable']:
        ret_dict['contrast_bright_limit'] = features['ContrastBrightLimit']
        ret_dict['contrast_dark_limit'] = features['ContrastDarkLimit']
    return ret_dict


def get_alliedvision_camera_model_name(camera):
    with camera as cam:
        model = cam.get_model()
        model = ''.join(map(lambda x: x.lower().capitalize(), model.replace(' ', '-').split('-')))
        return model
