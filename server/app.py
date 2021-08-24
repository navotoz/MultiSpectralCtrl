import logging
import multiprocessing as mp

import dash

from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheelCtrl import FilterWheelCtrl
from utils.logger import make_logging_handlers, make_logger
from utils.tools import make_duplex_pipe

logging.getLogger('werkzeug').disabled = True
app = dash.Dash(__name__, suppress_callback_exceptions=False, prevent_initial_callbacks=False)
server = app.server

LOGGING_LEVEL = logging.INFO
handlers = make_logging_handlers(None, True)
logger = make_logger('Server', handlers=handlers, level=LOGGING_LEVEL)

# FilterWheel
filterwheel = FilterWheelCtrl(logging_handlers=handlers)

# FLIR Tau2 Camera
image_storage = {}
_image_grabber_proc, image_grabber = make_duplex_pipe(flag_run=None)
flag_alive_camera = mp.Event()
flag_alive_camera.clear()
event_stop = mp.Event()  # This event signals all process in the program to stop
event_stop.clear()

camera = CameraCtrl(logging_handlers=handlers,
                    image_pipe=_image_grabber_proc,
                    event_stop=event_stop,
                    flag_alive=flag_alive_camera)
camera.start()
