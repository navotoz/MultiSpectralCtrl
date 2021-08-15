from pathlib import Path

FAILURE_PROBABILITY_IN_DUMMIES = 0.0
INIT_EXPOSURE = 50000  # micro seconds
SAVE_PATH = Path('download')
IMAGE_FORMAT = 'tiff'
DISPLAY_IMAGE_SIZE = '600px'
UPDATE_INTERVAL_SECONDS = 1
FILTERWHEEL_NAME = 'filterwheel'
FILTERWHEEL_POSITION = 'position'

TIFF_MODEL_NAME = 272
TIFF_EXPOSURE_TIME = 33434
TIFF_GAIN = 41991
TIFF_F_NUMBER = 33437
TIFF_FOCAL_LENGTH = 37386
TIFF_X_RESOLUTION = 282
TIFF_Y_RESOLUTION = 283
TIFF_NOTES= 37500

T_FPA = 'fpa'
T_HOUSING = 'housing'
CAMERA_TAU = 20
HEIGHT_IMAGE_TAU2 = 256
WIDTH_IMAGE_TAU2 = 336
WIDTH = 'width'
HEIGHT = 'height'
DIM = 'dim'
FFC_TEMPERATURE = 'ffc_temperature'
FFC = 'ffc'
FREQ_INNER_TEMPERATURE_SECONDS = 1
CAMERA_TAU_HERTZ = 0.01
CAMERA_NAME = "camera"


# device status
DEVICE_OFF = 0
DEVICE_DUMMY = 10
DEVICE_REAL = 100


CAMERA_PARAMETERS = 'camera_params'
INIT_CAMERA_PARAMETERS = dict(
    ffc_mode='external',
    isotherm=0x0000,
    dde=0x0000,
    tlinear=0x0000,
    gain='high',
    agc='manual',
    sso=0,
    contrast=0,
    brightness=0,
    brightness_bias=0,
    fps=0x0004,  # 60Hz   # TODO: CHANGE TO 30HZ (ADDS EXTRA LAYER OF AVG)
    lvds=0x0000,  # disabled
    lvds_depth=0x0000,  # 14bit
    xp=0x0002,  # 14bit w/ 1 discrete
    cmos_depth=0x0000,  # 14bit pre AGC
    # corr_mask=0  # off
)