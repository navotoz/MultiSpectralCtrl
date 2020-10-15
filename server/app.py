import dash
import logging
from utils.logger import make_logging_handlers, make_logger
from devices import valid_cameras_names_list, initialize_device

logging.getLogger('werkzeug').disabled = True
app = dash.Dash(__name__, suppress_callback_exceptions=False)
server = app.server

LOGGING_LEVEL = logging.INFO
handlers = make_logging_handlers(None, True, LOGGING_LEVEL)
logger = make_logger('Server', handlers=handlers, level=LOGGING_LEVEL)

image_store_dict = {}

filterwheel = initialize_device('filterwheel', handlers, use_dummy=True)
cameras_dict = dict.fromkeys(valid_cameras_names_list)
