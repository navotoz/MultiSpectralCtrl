import dash
import logging

from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheelThread import FilterWheelCtrl

from utils.logger import make_logging_handlers, make_logger
from utils.tools import make_duplex_pipe
import utils.constants as const

import multiprocessing as mp

logging.getLogger('werkzeug').disabled = True
app = dash.Dash(__name__, suppress_callback_exceptions=False, prevent_initial_callbacks=False)
server = app.server

LOGGING_LEVEL = logging.INFO
handlers = make_logging_handlers(None, True)
logger = make_logger('Server', handlers=handlers, level=LOGGING_LEVEL)
event_stop = mp.Event()  # This event signals all process in the program to stop
event_stop.clear()


# FilterWheel
filterwheel = FilterWheelCtrl(logging_handlers=handlers)
filterwheel.start()

# FLIR Tau2 Camera
image_storage = {}
_mp_manager = mp.Manager()
mp_values_dict = _mp_manager.dict({const.T_FPA: 0.0,
                                   const.T_HOUSING: 0.0
                                   })
_cam_cmd_proc, camera_cmd = make_duplex_pipe(flag_run=None)
_image_grabber_proc, image_grabber = make_duplex_pipe(flag_run=None)

camera = CameraCtrl(logging_handlers=handlers,
                    image_pipe=_image_grabber_proc,
                    event_stop=event_stop,
                    values_dict=mp_values_dict,
                    cmd_pipe=_cam_cmd_proc)
camera.start()
