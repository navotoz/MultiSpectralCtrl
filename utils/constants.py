from pathlib import Path

from devices.Camera import WIDTH_IMAGE_TAU2

FAILURE_PROBABILITY_IN_DUMMIES = 0.0
SAVE_PATH = Path('download')
DISPLAY_WIDTH = f'{2 * WIDTH_IMAGE_TAU2}px'
UPDATE_INTERVAL_SECONDS = 1
FILTERWHEEL_NAME = 'filterwheel'
FILTERWHEEL_POSITION = 'position'
N_IMAGE_INIT = 1

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
