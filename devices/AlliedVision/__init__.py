from importlib import import_module


def init_alliedvision_camera(model_name: (str, None), logging_handlers: (list, tuple), use_dummy: bool):
    if not use_dummy:
        from devices.AlliedVision.AlliedVisionCtrl import AlliedVisionCtrl as m
    else:
        from devices.AlliedVision.DummyAlliedVisionCtrl import AlliedVisionCtrl as m
    return m(model_name=model_name, logging_handlers=logging_handlers)


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


def get_specs_dict():
    return {
        '1800U1236m': dict(
            units='mm',
            h=4112,  # pixels
            w=3008,  # pixels
            # sensor='Sony IMX304',
            sensor_size_h=14.2,  # millimeter
            sensor_size_w=10.4,  # millimeter
            sensor_size_diag=17.6,  # millimeter
            # sensor_size=dict(h=14.2, w=10.4, diag=17.6),
            pixel_size=3.45e-3,  # millimeter
            focal_length=12,  # millimeter
            f_number=1.4,
            bit_depth=12  # bits
        ),
        'MakoU130b': dict(
            units='mm',
            h=1280,  # pixels
            w=1024,  # pixels
            # sensor='Sony IMX304',
            sensor_size_h=6.4,  # millimeter
            sensor_size_w=4.8,  # millimeter
            sensor_size_diag=8.0,  # millimeter
            # sensor_size=dict(h=14.2, w=10.4, diag=17.6),
            pixel_size=4.8e-3,  # millimeter
            focal_length=-1,  # millimeter
            f_number=-1,
            bit_depth=10  # bits
        )
    }


def get_features_dict():
    return {
        '1800U1236m': dict(
            exposure_min=178.79,  # microseconds
            exposure_max=9999983.0,  # microseconds
            exposure_increment=41.13,  # microseconds
            gain_min=0.0,  # dB
            gain_max=24.0,  # db
            gain_increment=0.1,  # db
            gamma_min=0.4,
            gamma_max=2.4,
            gamma_increment=0.05,
            autoexposure=True),
        'MakoU130b': dict(
            exposure_min=44.2,  # microseconds
            exposure_max=1.4e6,  # microseconds
            exposure_increment=10,  # microseconds   # todo: change to the real thing using the vimba viewer!!!
            gain_min=0.0,  # dB
            gain_max=20.0,  # db
            gain_increment=0.1,  # db
            gamma_min=0.4,
            gamma_max=2.4,
            gamma_increment=0.05,
            autoexposure=False)
    }
