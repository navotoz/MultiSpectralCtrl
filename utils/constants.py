from pathlib import Path

FAILURE_PROBABILITY_IN_DUMMIES = 0.0

INIT_EXPOSURE = 50000  # micro seconds
RECV_WAIT_TIME_IN_SEC = 0.5  # seconds
FILTER_WHEEL_SETTLING_TIME = 1  # seconds
DEFAULT_FILTER_NAMES_DICT = {1: '420', 2: '480', 3: '510', 4: '580', 5: '680', 6: '0'}  # 0 is glass

# SAVE_PATH = Path().cwd() / Path('download')
SAVE_PATH = Path('download')
IMAGE_FORMAT = 'tiff'


# FilterWheel commands
GET_ID = b'*idn?'
GET_POSITION = b'pos?'
GET_SPEED_MODE = b'speed?'
SET_SENSOR_MODE = b'sensors=0'
GET_SENSOR_MODE = b'sensors?'
GET_POSITION_COUNT = b'pcount?'
