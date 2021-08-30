import logging
from ctypes import c_ulong
from multiprocessing import Value

import dash

from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheelCtrl import FilterWheelCtrl
from utils.logger import make_logging_handlers, make_logger

logging.getLogger('werkzeug').disabled = True
app = dash.Dash(__name__, suppress_callback_exceptions=False, prevent_initial_callbacks=False)
server = app.server

LOGGING_LEVEL = logging.INFO
handlers = make_logging_handlers(None, True)
logger = make_logger('Server', handlers=handlers, level=LOGGING_LEVEL)

counter_images = Value(c_ulong)
counter_images.value = 1

# FilterWheel
filterwheel = FilterWheelCtrl(logging_handlers=handlers)

# FLIR Tau2 Camera
image_storage = {}
camera = CameraCtrl(logging_handlers=handlers)
camera.start()
