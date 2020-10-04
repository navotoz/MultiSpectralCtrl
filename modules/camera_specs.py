CAMERAS_SPECS_DICT = dict(
    Alvium1800u1236=dict(
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
    ),
    MakoU130b=dict(
        units='mm',
        h=1280,  # pixels
        w=1024,  # pixels
        # sensor='Sony IMX304',
        sensor_size_h=6.4,  # millimeter
        sensor_size_w=4.8,  # millimeter
        sensor_size_diag=8.0,  # millimeter
        # sensor_size=dict(h=14.2, w=10.4, diag=17.6),
        pixel_size=4.8e-3,  # millimeter
        bit_depth=10  # bits
    )
)


CAMERAS_FEATURES_DICT = dict(
    Alvium1800u1236=dict(
        exposure_min=178.79,   # microseconds
        exposure_max=9999983.0,  # microseconds
        exposure_increment=41.13,  # microseconds
        gain_min=0.0,  # dB
        gain_max=24.0,  # db
        gain_increment=0.1,  # db
        gamma_min=0.4,
        gamma_max=2.4,
        gamma_increment=0.05),
# MakoU130b = dict(
#     exposure_min=,  # microseconds
#     exposure_max=,  # microseconds
#     exposure_increment=,  # microseconds
#     gain_min=,  # dB
#     gain_max=,  # db
#     gain_increment=,  # db
#     gamma_min=,
#     gamma_max=,
#     gamma_increment=)
)
