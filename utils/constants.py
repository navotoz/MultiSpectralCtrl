from pathlib import Path

from devices.Camera.Tau.tau2_config import FPS_CODE_DICT

FAILURE_PROBABILITY_IN_DUMMIES = 0.0
SAVE_PATH = Path('download')
HEIGHT_IMAGE_TAU2 = 256
WIDTH_IMAGE_TAU2 = 336
DISPLAY_WIDTH = f'{2*WIDTH_IMAGE_TAU2}px'
UPDATE_INTERVAL_SECONDS = 1
FILTERWHEEL_NAME = 'filterwheel'
FILTERWHEEL_POSITION = 'position'
N_IMAGE_INIT = 1

T_FPA = 'fpa'
T_HOUSING = 'housing'
CAMERA_TAU = 20
WIDTH = 'width'
HEIGHT = 'height'
DIM = 'dim'
FFC_TEMPERATURE = 'ffc_temperature'
FFC = 'ffc'
FREQ_INNER_TEMPERATURE_SECONDS = 30
CAMERA_NAME = "camera"

# device status
DEVICE_OFF = 0
DEVICE_DUMMY = 10
DEVICE_REAL = 100

CAMERA_PARAMETERS = 'camera_params'
INIT_CAMERA_PARAMETERS = dict(
    ffc_mode='auto',
    isotherm=0x0000,
    dde=0x0000,
    tlinear=0x0000,
    gain='high',
    agc='manual',
    sso=0,
    ace=0,
    contrast=0,
    brightness=0,
    brightness_bias=0,
    fps=FPS_CODE_DICT[30],
    lvds=0x0000,  # disabled
    lvds_depth=0x0000,  # 14bit
    xp=0x0002,  # 14bit w/ 1 discrete
    cmos_depth=0x0000,  # 14bit pre AGC
    # corr_mask=0  # off
)
