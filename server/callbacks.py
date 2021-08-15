from flask import request
import os
from io import BytesIO
from pathlib import Path
from typing import Dict

import dash
from dash.dependencies import Input, Output, State
from flask import Response, send_file

from server.app import app, server, logger, handlers, filterwheel, camera
from server.tools import find_files_in_savepath, save_image_to_tiff, base64_to_split_numpy_image, only_numerics
from server.tools import make_images_for_web_display, make_links_from_files
from utils.constants import SAVE_PATH, IMAGE_FORMAT
import dash_html_components as html
from threading import Event, Lock
from utils.logger import dash_logger

image_store_dict = dict()
dict_flags_change_camera_mode: Dict[str, Event or Lock] = dict().fromkeys(["Tau2"], Event())
event_finished_image = Event()
event_finished_image.set()


@server.route("/download/<path:path>")
def download(path: (str, Path)) -> Response:
    """Serve a file from the upload directory."""
    file_stream = BytesIO()
    with open(SAVE_PATH / str(path), 'rb') as fp:
        file_stream.write(fp.read())
    file_stream.seek(0)
    os.remove(SAVE_PATH / str(path))
    return send_file(file_stream, as_attachment=True, attachment_filename=path)


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
def show_images(trigger1, trigger2):
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
    logger.debug('Showing images.')
    return table_cells_list


@app.callback([Output('take-photo-button', 'disabled')],
              [Input('take-photo-button', 'n_clicks'),
               Input('interval-component', 'n_intervals')])
def disable_button(dummy1, dummy2):
    callback_trigger = dash.callback_context.triggered[0]['prop_id']
    global event_finished_image
    if not event_finished_image.is_set():
        return dash.no_update
    if 'take-photo-button' in callback_trigger:
        event_finished_image.clear()
        return True,
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
               State(f"image-sequence-length", 'value')])
def images_handler_callback(button_state, to_save: str, length_sequence: int):
    if button_state:
        global event_finished_image
        event_finished_image.clear()
        global image_store_dict
        global cameras_dict
        global filterwheel
        image_store_dict = dict()
        camera_names_list = filter(lambda name: cameras_dict[name], cameras_dict.keys())
        camera_names_list = list(filter(lambda name: multispectral_camera_name not in name, camera_names_list))
        if not multispectral_camera_name:
            event_finished_image.set()
            return 1

        # take images for different filters
        logger.info(f"Taking a sequence of {length_sequence} filters.")
        for position in range(1, length_sequence + 1):
            for camera_name in camera_names_list:  # photo with un-filtered cameras
                image_store_dict.setdefault(camera_name, []).append(cameras_dict[camera_name]())
            filterwheel.position = position
            image = cameras_dict[multispectral_camera_name]()
            if image is not None:
                image_store_dict.setdefault(multispectral_camera_name, []).append((filterwheel.position['name'], image))
            logger.info(f"Taken an image in position {position}#.")

        # get specs for all cameras
        for camera_name in camera_names_list + [multispectral_camera_name]:
            image_store_dict.setdefault(camera_name, []).append(cameras_dict[camera_name].parse_specs_to_tiff())

        # save images (if to_save)
        for camera_name in camera_names_list + [multispectral_camera_name]:  # photo with un-filtered cameras
            save_image_to_tiff(image_store_dict[camera_name]) if to_save else None
        logger.info("Taken a sequence." if 'save' not in to_save else "Saved a sequence.")
        event_finished_image.set()
        return 1
    return dash.no_update


@app.callback(Output('file-list', 'n_clicks'),
              [Input('upload-img-button', 'contents')],
              [State('upload-img-button', 'filename')])
def upload_image(content, name):
    if content is not None:
        path = Path(name)
        if IMAGE_FORMAT not in path.suffix.lower():
            logger.error(f"Given image suffix is {path.suffix}, different than required {IMAGE_FORMAT}.")
            return 1
        global image_store_dict
        num_of_filters = int(path.stem.split('Filters')[0].split('_')[-1])
        filters_names = path.stem.split('Filters')[-1].split('_')[-num_of_filters:]
        image = base64_to_split_numpy_image(content)
        image_store_dict = {key: val for key, val in zip(filters_names, image)}
        logger.info(f"Uploaded {len(image_store_dict.keys())} frames.")
    return 1


@app.callback(Output('filter-names-label', 'n_clicks'),
              [Input('image-sequence-length-label', 'n_clicks')] +
              [Input(f"filter-{idx}", 'n_submit') for idx in range(1, filterwheel.position_count + 1)] +
              [Input(f"filter-{idx}", 'n_blur') for idx in range(1, filterwheel.position_count + 1)],
              [State(f"filter-{idx}", 'value') for idx in range(1, filterwheel.position_count + 1)])
def change_filter_names(*args):
    global filterwheel
    position_names_dict = dict(zip(range(1, filterwheel.position_count + 1), args[-filterwheel.position_count:]))
    if filterwheel.position_names_dict != position_names_dict:
        filterwheel.position_names_dict = position_names_dict
    return 1


@app.callback(Output('kill-button', 'children'),
              Input('kill-button', 'n_clicks'))
def kill_server(n_clicks):
    if not n_clicks:
        return dash.no_update
    exit_handler(None, None)
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    exit(0)


def exit_handler(sig_type: int, frame) -> None:
    camera.__del__()
    exit(0)


@app.callback(
    Output('log-div', 'children'),
    Input('interval-component', 'n_intervals'))
def log_content(n_intervals):
    if dash_logger.dirty_bit:
        ret = [html.Div(([html.Div(children=[html.Div(l) for l in log],
                                   style=dict(height='200px', overflow='auto'))] + [html.Hr()])) for log in
               dash_logger.logs.values()]
        dash_logger.dirty_bit = False
        return ret
    return dash.no_update


@app.callback(
    [Output('filterwheel-status', 'style'),
    Output('filterwheel-status', 'children')],
    Input('interval-component', 'n_intervals'),
    State('filterwheel-status', 'style'))
def check_valid_filterwheel(n_intervals, style):
    if n_intervals:
        if filterwheel.is_dummy:
            style['background'] = None
            return style, 'Dummy'
        else:
            style['background'] = 'green'
            return style, 'Real'
    return dash.no_update


@app.callback(
    [Output('tau2-status', 'style'),
    Output('tau2-status', 'children')],
    Input('interval-component', 'n_intervals'),
    State('tau2-status', 'style'))
def check_valid_filterwheel(n_intervals, style):
    if n_intervals:
        if camera.is_dummy:
            style['background'] = None
            return style, 'Dummy'
        else:
            style['background'] = 'green'
            return style, 'Real'
    return dash.no_update