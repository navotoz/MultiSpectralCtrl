import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
from pathlib import Path

from utils.camera_specs import CAMERAS_FEATURES_DICT
from utils.logger import make_logging_handlers, make_logger

from flask import Flask, Response, send_file
from server.server_utils import make_values_dict, save_image, make_images, make_links_from_files
from server.server_utils import base64_to_split_numpy_image, find_files_in_savepath
from server.layout import main_layout

### To run on test mode without camera import from test.dummMultiFrame
# from MultiFrame import MultiFrameGrabber
from tests.dummy_MultiFrame import MultiFrameGrabber


from utils.constants import DEFAULT_FILTER_NAMES_DICT, SAVE_PATH, IMAGE_FORMAT
import os
from io import BytesIO
import logging
import json

logging.getLogger('werkzeug').disabled = True

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div(id='layout', children=main_layout)


@server.route("/download/<path:path>")
def download(path: (str, Path)) -> Response:
    """Serve a file from the upload directory."""
    file_stream = BytesIO()
    with open(SAVE_PATH / str(path), 'rb') as fp:
        file_stream.write(fp.read())
    file_stream.seek(0)
    os.remove(SAVE_PATH / str(path))
    return send_file(file_stream, as_attachment=True, attachment_filename=path)


@app.callback(Output('input_exposure_time', 'disabled'), [Input('exposure_type_radio', 'value')])
def is_exposure_auto(exp_type):
    if exp_type == 'auto':
        grabber.auto_exposure = True
        return True
    if exp_type == 'manual':
        grabber.auto_exposure = False
        return False


@app.callback([Output('input_exposure_time', 'value'), Output('input_exposure_time', 'min'),
               Output('input_exposure_time', 'max'), Output('input_exposure_time', 'step'),
               Output('input_gain_time', 'min'), Output('input_gain_time', 'max'), Output('input_gain_time', 'step'),
               Output('input_gamma_time', 'min'), Output('input_gamma_time', 'max'),
               Output('input_gamma_time', 'step')],
              [Input('choose_camera_model', 'value')])
def change_values_limits(model_name):
    return make_values_dict(CAMERAS_FEATURES_DICT, model_name)


@app.callback(Output('focal_tag', 'n_clicks'),
              [Input('choose_camera_model', 'value'),
               Input('input_exposure_time', 'value'),
               Input('input_gain_time', 'value'),
               Input('input_gamma_time', 'value'),
               Input('input_lens_focal_length_mm', 'value'),
               Input('input_lens_f_number', 'value')
               ])
def update_values_in_camera(camera_model, exposure_time_value, gain_value, gamma_value, focal_length, f_num):
    grabber.camera_model = camera_model
    grabber.focal_length = focal_length
    grabber.f_number = f_num
    grabber.gain = gain_value
    grabber.exposure_time = exposure_time_value
    grabber.gamma = gamma_value
    log.debug(f"Updated camera values.")
    return 1


@app.callback(Output('filter_names_div', 'n_clicks'),
              [Input(f"filter_{idx}", 'n_submit') for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)] +
              [Input(f"filter_{idx}", 'n_blur') for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)],
              [State(f"filter_{idx}", 'value') for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)])
def change_filter_names(*args):
    grabber.filter_names_dict = dict(zip(range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1),
                                         args[-len(DEFAULT_FILTER_NAMES_DICT):]))
    log.debug(f"Changed filters names.")
    return 1


@app.callback(Output('image_seq_len_div', 'n_clicks'),
              [Input(f"image_sequence_length", 'value')])
def set_image_sequence_length(image_sequence_length):
    grabber.filters_sequence = list(range(1, image_sequence_length + 1))
    log.debug(f"Set filter sequence length.")
    return 1


@app.callback(Output('after_take_photo', 'n_clicks'),
              [Input('take_photo_button', 'disabled')],
              [State('save_img_checkbox', 'value')])
def images_handler_callback(button_state, handler_flags):
    if button_state:
        global image_store_dict
        image_store_dict = save_image(grabber, 'save_img' in handler_flags)
        log.info("Taken an image." if 'save_img' not in handler_flags else "Saved an image.")
        return 1
    return dash.no_update


@app.callback([Output('take_photo_button', 'disabled')],
              [Input('take_photo_button', 'n_clicks'), Input('after_take_photo', 'n_clicks')],
              [State('take_photo_button', 'disabled')])
def disable_button(n_clicks, trigger, button_state):
    """
    Controls the 'Take Image' button disable status.
    Can be triggered by pressing the button or by images_handler_callback.

    Args:
        n_clicks:
        trigger:
        button_state:

    Returns:
        disables the button if the trigger came from images_handler_callback, else enables the button.
    """
    callback_trigger = dash.callback_context.triggered[0]['prop_id']
    if 'after_take_photo.n_clicks' not in callback_trigger and n_clicks > 0 and not button_state:
        log.debug(f"Disabled the image button.")
        return True,
    log.debug(f"Enabled the image button.")
    return False,


@app.callback(Output('file_list', 'children'),
              [Input(f"after_take_photo", 'n_clicks'), Input('file_list', 'n_clicks')])
def make_downloads_list(dummy1, dummy2):
    """
    Make a list of files in SAVE_PATH with type defined in utils.constants.
    The list is transformed into a list of links and is displayed on the webpage.
    This function is called during the initial loading of the page, and after every button press.

    Returns:
        list: links to the files in SAVE_PATH of a predefined filetype.
    """
    file_list_images = find_files_in_savepath()
    links_list = make_links_from_files(file_list_images)
    log.debug(f"Found {len(links_list)} files to download.")
    return links_list


@app.callback(Output('file_list', 'n_clicks'),
              [Input('upload_img', 'contents')],
              [State('upload_img', 'filename')])
def upload_image(content, name):
    if content is not None:
        path = Path(name)
        if IMAGE_FORMAT not in path.suffix:
            log.error(f"Given image suffix is {path.suffix}, different than required {IMAGE_FORMAT}.")
            return 1
        global image_store_dict
        num_of_filters = int(path.stem.split('Filters')[0].split('_')[-1])
        filters_names = path.stem.split('Filters')[-1].split('_')[-num_of_filters:]
        image = base64_to_split_numpy_image(content, H_IMAGE, W_IMAGE)
        image_store_dict = {key: val for key, val in zip(filters_names, image)}
        log.info(f"Uploaded {len(image_store_dict.keys())} frames.")
    return 1


@app.callback(Output("imgs", 'children'),
              [Input('file_list', 'n_clicks'), Input('after_take_photo', 'n_clicks')])
def show_images(dummy1, dummy2):
    global image_store_dict
    bboxs = make_images(image_store_dict)
    log.debug('Showing images.')
    return bboxs


image_store_dict = {}
if __name__ == '__main__':
    handlers = make_logging_handlers(None, True)
    grabber = MultiFrameGrabber(0, 0, handlers, dummy_filterwheel=True)
    log = make_logger('Server', handlers=handlers)
    log.setLevel(logging.INFO)
    H_IMAGE = grabber.camera_specs.get('h')
    W_IMAGE = grabber.camera_specs.get('w')
    PORT = 8000
    # IP = "127.0.0.1"
    IP = "0.0.0.0"
    print(f"http://{IP:s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=False, host=IP, port=PORT, threaded=True)


# todo: multiple cameras - should try to run as many cameras as possible. When detecting a camera, her page becomes available.
## "take a photo" takes a photo in all cameras at the same time.