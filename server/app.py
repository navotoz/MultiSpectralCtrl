import dash
import logging
from utils.logger import make_logging_handlers, make_logger
from devices import get_alliedvision_grabber
PATHNAME_MAIN = '/apps/main'
PATHNAME_INIT_DEVICES = '/apps/init'

logging.getLogger('werkzeug').disabled = True
app = dash.Dash(__name__, suppress_callback_exceptions=False)
server = app.server

handlers = make_logging_handlers(None, True)
logger = make_logger('Server', handlers=handlers)
logger.setLevel(logging.INFO)

image_store_dict = {}

filterwheel = None
grabber = get_alliedvision_grabber(use_dummy_alliedvision_camera=True,
                                   use_dummy_filterwheel=True,
                                   focal_length_mm=0, f_number=0, logging_handlers=handlers,
                                   camera_model='ALVIUM_1800U_1236')