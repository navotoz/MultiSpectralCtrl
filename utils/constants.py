from pathlib import Path

INIT_EXPOSURE = 50000  # micro seconds
RECV_WAIT_TIME_IN_SEC = 0.5  # seconds
FILTER_WHEEL_SETTLING_TIME = 1  # seconds
DEFUALT_FOCAL_LENGTH = 12  # millimeters
DEFUALT_F_NUMBER = 1.4
DEFAULT_FILTER_NAMES_DICT = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f'}

# SAVE_PATH = Path().cwd() / Path('images')
SAVE_PATH = Path('images')
IMAGE_FORMAT = 'tiff'
