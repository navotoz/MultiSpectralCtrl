CAMERAS_SPECS_DICT = dict(
    ALVIUM_1800U_1236=dict(
        units='mm',
        h=4112,  # pixels
        w=3008,  # pixels
        # sensor='Sony IMX304',
        sensor_size_h=14.2,  # millimeter
        sensor_size_w=10.4,  # millimeter
        sensor_size_diag=17.6,  # millimeter
        # sensor_size=dict(h=14.2, w=10.4, diag=17.6),
        pixel_size=3.45e-3,  # millimeter
        bit_depth=12  # bits
    )
)

CAMERAS_FEATURES_DICT = dict(
    ALVIUM_1800U_1236=dict(
        exposure_min=178.79,   # microseconds
        exposure_max=9999983.0,  # microseconds
        exposure_increment=41.13,  # microseconds
        gain_min=0.0,  # dB
        gain_max=24.0,  # db
        gain_increment=0.1,  # db
        gamma_min=0.4,
        gamma_max=2.4,
        gamma_increment=0.05)
)
