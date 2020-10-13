from importlib import import_module


def init_ids_camera(model_name: (str, None), logging_handlers: (list, tuple), use_dummy: bool):
    use_dummy = 'Dummy' if use_dummy else ''
    path = "devices.IDS"
    m = import_module(f"{path}.{use_dummy}IDSCtrl", f"{use_dummy}IDSCtrl").IDSCtrl
    return m(model_name=model_name, logging_handlers=logging_handlers)


def get_specs_dict():
    return {
        'Ui314xcpC': dict(
            units='mm',
            h=1280,  # pixels
            w=1024,  # pixels
            sensor_size_h=6.144,  # millimeter
            sensor_size_w=4.915,  # millimeter
            sensor_size_diag=7.87,  # millimeter
            # sensor_size=dict(h=14.2, w=10.4, diag=17.6),
            pixel_size=4.8e-3,  # millimeter
            focal_length=12,  # millimeter
            f_number=1.4,
            bit_depth=12  # bits
        )
    }


def get_features_dict():
    return {
        'Ui314xcpC': dict(
            exposure_min=178.79,  # microseconds
            exposure_max=9999983.0,  # microseconds
            exposure_increment=41.13,  # microseconds
            gain_min=0.0,  # dB
            gain_max=24.0,  # db
            gain_increment=0.1,  # db
            gamma_min=0.4,
            gamma_max=2.4,
            gamma_increment=0.05,
            autoexposure=True)
    }
