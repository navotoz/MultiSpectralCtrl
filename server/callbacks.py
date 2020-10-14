import os
from io import BytesIO
from pathlib import Path

import dash
from dash.dependencies import Input, Output, State
from flask import Response, send_file

from devices import initialize_device, valid_cameras_names_list
from devices import SPECS_DICT, FEATURES_DICT
from server.app import app, server, logger, handlers, filterwheel, cameras_dict
from server.utils import find_files_in_savepath, save_image_to_tiff, get_filters_tags_images
from server.utils import make_images_for_web_display, make_links_from_files, make_models_dropdown_options_list
from utils.constants import SAVE_PATH
import dash_html_components as html

# H_IMAGE = grabber.camera_specs.get('h')
# W_IMAGE = grabber.camera_specs.get('w')

image_store_dict = {}


@server.route("/download/<path:path>")
def download(path: (str, Path)) -> Response:
    """Serve a file from the upload directory."""
    file_stream = BytesIO()
    with open(SAVE_PATH / str(path), 'rb') as fp:
        file_stream.write(fp.read())
    file_stream.seek(0)
    os.remove(SAVE_PATH / str(path))
    return send_file(file_stream, as_attachment=True, attachment_filename=path)


@app.callback([Output('focal-length', 'value'), Output('f-number', 'value')],
              [Input('camera-model-dropdown', 'value')])
def update_optical_values(camera_model_name: str):
    if not camera_model_name:
        return dash.no_update
    return SPECS_DICT[camera_model_name].get('focal_length', 0), \
           SPECS_DICT[camera_model_name].get('f_number', 0)


@app.callback([Output('exposure-type-radio', 'options'),
               Output('exposure-time', 'min'),
               Output('exposure-time', 'max'),
               Output('exposure-time', 'step')],
              [Input('camera-model-dropdown', 'value')])
def update_exposure(camera_model_name: str):
    if not camera_model_name:
        return dash.no_update
    exposure_options_list = [{'label': 'Manual', 'value': 'manual'}]
    if FEATURES_DICT[camera_model_name].get('autoexposure', False):
        exposure_options_list.append({'label': 'Auto', 'value': 'auto'})
    return exposure_options_list, \
           FEATURES_DICT[camera_model_name].get('exposure_min'), \
           FEATURES_DICT[camera_model_name].get('exposure_max'), \
           FEATURES_DICT[camera_model_name].get('exposure_increment')


@app.callback([Output('gain', 'min'),
               Output('gain', 'max'),
               Output('gain', 'step'),
               Output('gamma', 'min'),
               Output('gamma', 'max'),
               Output('gamma', 'step')],
              [Input('camera-model-dropdown', 'value')])
def update_gain_gamma(camera_model_name: str):
    if not camera_model_name:
        return dash.no_update
    return FEATURES_DICT[camera_model_name].get('gain_min'), \
           FEATURES_DICT[camera_model_name].get('gain_max'), \
           FEATURES_DICT[camera_model_name].get('gain_increment'), \
           FEATURES_DICT[camera_model_name].get('gamma_min'), \
           FEATURES_DICT[camera_model_name].get('gamma_max'), \
           FEATURES_DICT[camera_model_name].get('gamma_increment')


@app.callback(Output('exposure-time', 'disabled'),
              [Input('exposure-type-radio', 'value'),
               Input('camera-model-dropdown', 'value')])
def set_auto_exposure(exposure_type, camera_model_name):
    if not camera_model_name:
        return True
    if exposure_type == 'auto':
        cameras_dict[camera_model_name].exposure_auto = True
        return True
    if exposure_type == 'manual':
        cameras_dict[camera_model_name].exposure_auto = False
        return False


@app.callback([Output('gain', 'disabled'),
               Output('gamma', 'disabled'),
               Output('focal-length', 'disabled'),
               Output('f-number', 'disabled')],
              [Input('camera-model-dropdown', 'value')])
def set_disabled_to_camera_values(camera_model_name):
    if not camera_model_name:
        return [True] * len(dash.callback_context.outputs_list)
    return [False] * len(dash.callback_context.outputs_list)


@app.callback(Output('focal-length-label', 'n_clicks'),
              [Input('gain', 'value'),
               Input('gamma', 'value'),
               Input('exposure-time', 'value'),
               Input('camera-model-dropdown', 'value')])
def update_values_in_camera(gain, gamma, exposure_time, camera_model_name):
    if not camera_model_name:
        return dash.no_update
    global cameras_dict
    cameras_dict[camera_model_name].gain = gain
    cameras_dict[camera_model_name].gamma = gamma
    cameras_dict[camera_model_name].exposure_time = exposure_time
    logger.debug(f"Updated camera values.")
    return 1


@app.callback(Output('file-list', 'children'),
              [Input(f"after-photo-sync-label", 'n_clicks'),
               Input('file-list', 'n_clicks')])
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
    logger.debug(f"Found {len(links_list)} files to download.")
    return links_list


@app.callback(Output("imgs", 'children'),
              [Input('file-list', 'n_clicks'),
               Input('after-photo-sync-label', 'n_clicks')])
def show_images(dummy1, dummy2):
    global image_store_dict
    if not image_store_dict:
        return dash.no_update
    camera_names_list = list(image_store_dict.keys())
    ms_name = list(filter(lambda name: isinstance(image_store_dict[name][0], tuple), camera_names_list))[-1]
    camera_names_list.remove(ms_name)
    camera_names_list.insert(0, ms_name)
    table_cells_list = [html.Tr([html.Td(name) for name in camera_names_list])]
    num_of_filters = len(image_store_dict[ms_name]) - 1  # -1 for the spec dict in the end
    for idx in range(num_of_filters):
        image_list = []
        for camera_name in camera_names_list:
            image = image_store_dict[camera_name][idx]
            image_list.append(image if isinstance(image, tuple) else ('0', image))
        table_cells_list.append(make_images_for_web_display(image_list))
    table = html.Table(html.Tr(children=[*table_cells_list]))
    logger.debug('Showing download.')
    return table


def check_device_state(name: str) -> str:
    if not cameras_dict[name]:
        return 'none'
    if cameras_dict[name].is_dummy:
        return 'dummy'
    return 'real'


def is_equal_states(context_list):
    radioitems_states = list(map(lambda state: (state['id'].split('-')[0], state['value'].lower()), context_list))
    devices_state = list(map(lambda name: check_device_state(name), cameras_dict.keys()))
    if all([dev == radio[-1] for dev, radio in zip(devices_state, radioitems_states)]):
        return True, radioitems_states
    return False, radioitems_states


@app.callback([Output('devices-radioitems-table', 'n_clicks'), ],
              [Input(f'{name}-camera-type-radio', 'value') for name in valid_cameras_names_list])
def change_camera_status(*args):
    global cameras_dict
    states_equal, radioitems_states = is_equal_states(dash.callback_context.inputs_list)
    if states_equal:
        return dash.no_update
    for name, state in radioitems_states:
        if 'none' in state:
            if cameras_dict[name] and name in dash.callback_context.triggered[0]['prop_id']:
                logger.info(f'{name} camera is not used.')
            cameras_dict[name] = None
        elif 'dummy' in state and (not cameras_dict[name] or check_device_state(name)):
            cameras_dict[name] = initialize_device(name, handlers, use_dummy=True)
        elif 'real' in state:
            try:
                cameras_dict[name] = initialize_device(name, handlers, use_dummy=False)
            except RuntimeError:
                cameras_dict[name] = None
    return 1,


@app.callback([Output(f'{name}-camera-type-radio', 'value') for name in valid_cameras_names_list],
              [Input('devices-radioitems-table', 'n_clicks')],
              [State(f'{name}-camera-type-radio', 'value') for name in valid_cameras_names_list])
def update_devices_radiobox(*args):
    if not args[0] or is_equal_states(dash.callback_context.states_list)[0]:
        return dash.no_update
    return list(map(lambda name: check_device_state(name), cameras_dict.keys()))


@app.callback([Output('camera-model-dropdown', 'options'),
               Output('multispectral-camera-radioitems', 'options')],
              Input('devices-radioitems-table', 'n_clicks'))
def update_camera_models_dropdown_list(dummy):
    device_state_list = list(map(lambda name: (name, check_device_state(name)), cameras_dict.keys()))
    dropdown_options = make_models_dropdown_options_list(device_state_list)
    return dropdown_options, dropdown_options


@app.callback(Output('multispectral-camera-radioitems', 'value'),
              Input('multispectral-camera-radioitems', 'options'))
def update_camera_models_dropdown_list(dropdown_options):
    ms_name = dropdown_options[0]['value'] if dropdown_options else None
    return ms_name


@app.callback([Output('take-photo-button', 'disabled')],
              [Input('take-photo-button', 'n_clicks'),
               Input('after-photo-sync-label', 'n_clicks')],
              [State('take-photo-button', 'disabled')])
def disable_button(n_clicks, dummy, button_state):
    callback_trigger = dash.callback_context.triggered[0]['prop_id']
    if 'after-photo-sync-label' not in callback_trigger and n_clicks > 0 and not button_state:
        logger.debug(f"Disabled the image button.")
        return True,
    logger.debug(f"Enabled the image button.")
    return False,


@app.callback(Output('image-sequence-length-label', 'n_clicks'),
              Input(f"image-sequence-length", 'value'))
def set_image_sequence_length(image_sequence_length: int):
    filter_sequence = list(range(1, image_sequence_length + 1))
    logger.debug(f"Set filter sequence length to {len(filter_sequence)}.")
    return 1


@app.callback(Output('after-photo-sync-label', 'n_clicks'),
              Input('take-photo-button', 'disabled'),
              [State('save-image-checkbox', 'value'),
               State('multispectral-camera-radioitems', 'value'),
               State(f"image-sequence-length", 'value')])
def images_handler_callback(button_state, to_save: str, multispectral_camera_name: str, length_sequence: int):
    if button_state:
        global image_store_dict
        global cameras_dict
        global filterwheel
        image_store_dict = dict()
        camera_names_list = filter(lambda name: cameras_dict[name], cameras_dict.keys())
        camera_names_list = list(filter(lambda name: multispectral_camera_name not in name, camera_names_list))
        if not multispectral_camera_name:
            return 1

        # take images for different filters
        for position in range(1, length_sequence + 1):
            for camera_name in camera_names_list:  # photo with un-filtered cameras
                image_store_dict.setdefault(camera_name, []).append(cameras_dict[camera_name]())
            filterwheel.position = position
            image = cameras_dict[multispectral_camera_name]()
            image_store_dict.setdefault(multispectral_camera_name, []).append((filterwheel.position['name'], image))

        # get specs for all cameras
        for camera_name in camera_names_list + [multispectral_camera_name]:
            image_store_dict.setdefault(camera_name, []).append(cameras_dict[camera_name].parse_specs_to_tiff())

        # save images (if to_save)
        for camera_name in camera_names_list + [multispectral_camera_name]:  # photo with un-filtered cameras
            save_image_to_tiff(image_store_dict[camera_name]) if to_save else None
        logger.info("Taken an image." if 'save' not in to_save else "Saved an image.")
        return 1
    return dash.no_update



# @app.callback(Output('file-list', 'n_clicks'),
#               [Input('upload-img-button', 'contents')],
#               [State('upload-img-button', 'filename')])
# def upload_image(content, name):
#     if content is not None:
#         path = Path(name)
#         if IMAGE_FORMAT not in path.suffix:
#             logger.error(f"Given image suffix is {path.suffix}, different than required {IMAGE_FORMAT}.")
#             return 1
#         global image_store_dict
#         num_of_filters = int(path.stem.split('Filters')[0].split('_')[-1])
#         filters_names = path.stem.split('Filters')[-1].split('_')[-num_of_filters:]
#         image = base64_to_split_numpy_image(content, H_IMAGE, W_IMAGE)
#         image_store_dict = {key: val for key, val in zip(filters_names, image)}
#         logger.info(f"Uploaded {len(image_store_dict.keys())} frames.")
#     return 1


@app.callback(Output('use-real-filterwheel-midstep', 'children'),
              Input('use-real-filterwheel', 'value'),
              State('use-real-filterwheel-midstep', 'children'))
def get_real_filterwheel_midstep(value, next_value):
    if value and isinstance(value, list) or isinstance(value, tuple):
        value = value[0]
    if value == next_value:
        return dash.no_update
    return value


@app.callback([Output('use-real-filterwheel', 'value'), ],
              [Input('use-real-filterwheel-midstep', 'children')])
def get_real_filterwheel(value: str):
    global filterwheel
    if not value:  # use the dummy
        if not filterwheel.is_dummy:
            filterwheel = initialize_device('FilterWheel', handlers, use_dummy=True)
        return (),
    else:  # use the real FilterWheel
        try:
            filterwheel = initialize_device('FilterWheel', handlers, use_dummy=False)
        except RuntimeError:
            filterwheel = initialize_device('FilterWheel', handlers, use_dummy=True)
            return [],
        return [value],


@app.callback(Output('filter-names-label', 'n_clicks'),
              [Input('image-sequence-length-label', 'n_clicks')]+
              [Input(f"filter-{idx}", 'n_submit') for idx in range(1, filterwheel.position_count + 1)] +
              [Input(f"filter-{idx}", 'n_blur') for idx in range(1, filterwheel.position_count + 1)],
              [State(f"filter-{idx}", 'value') for idx in range(1, filterwheel.position_count + 1)])
def change_filter_names(*args):
    global filterwheel
    position_names_dict = dict(zip(range(1, filterwheel.position_count + 1), args[-filterwheel.position_count:]))
    if filterwheel.position_names_dict != position_names_dict:
        filterwheel.position_names_dict = position_names_dict
    return 1