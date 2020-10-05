import dash
import logging
from utils.logger import make_logging_handlers, make_logger
from devices.CamerasCtrl import valid_model_names_list
PATHNAME_MAIN = '/apps/main'
PATHNAME_INIT_DEVICES = '/apps/init'

logging.getLogger('werkzeug').disabled = True
app = dash.Dash(__name__, suppress_callback_exceptions=False)
server = app.server

LOGGING_LEVEL = logging.DEBUG
handlers = make_logging_handlers(None, True, LOGGING_LEVEL)
logger = make_logger('Server', handlers=handlers, level=LOGGING_LEVEL)

image_store_dict = {}

filterwheel= None
cameras_dict = dict.fromkeys(valid_model_names_list)