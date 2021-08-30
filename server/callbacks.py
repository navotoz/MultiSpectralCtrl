import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Event, Lock
from typing import Dict

import dash
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from flask import request, Response, send_file

import utils.constants as const
# noinspection PyUnresolvedReferences
from server.app import app, server, logger, camera, filterwheel, counter_images
from server.tools import find_files_in_savepath, base64_to_split_numpy_image, make_images_for_web_display, \
    make_links_from_files
from utils.constants import SAVE_PATH, IMAGE_FORMAT
from utils.logger import dash_logger

image_storage = list()
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


@app.callback([Output('file-list-npy', 'children'),
               Output('file-list-csv', 'children')],
              Input('interval-component', 'n_intervals'))
def make_downloads_list(interval):
    """
    Make a list of files in SAVE_PATH with type defined in utils.constants.
    The list is transformed into a list of links and is displayed on the webpage.
    This function is called during the initial loading of the page, and after every button press.

    Returns:
        list: links to the files in SAVE_PATH of a predefined filetype.
    """
    file_list_images = find_files_in_savepath('npy')
    links_list_images = make_links_from_files(file_list_images)
    file_list_csv = find_files_in_savepath('csv')
    links_list_csv = make_links_from_files(file_list_csv)
    return links_list_images, links_list_csv


@app.callback(Output("imgs", 'children'),
              [Input('file-list', 'n_clicks'),
               Input('after-photo-sync-label', 'n_clicks')])
def show_images(trigger1, trigger2):
    global image_storage
    if not image_storage:
        return dash.no_update
    table_cells_list = []
    table_cells_list.append(make_images_for_web_display(list(image_storage.items())))
    logger.debug('Showing images.')
    return table_cells_list


@app.callback([Output('take-photo-button', 'disabled')],
              [Input('take-photo-button', 'n_clicks'),
               Input('interval-component', 'n_intervals')])
def disable_button(dummy1, dummy2):
    callback_trigger = dash.callback_context.triggered[0]['prop_id']
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


def exit_handler(sig_type: int, frame) -> None:
    try:
        camera.__del__()
    except (ValueError, TypeError, AttributeError, RuntimeError):
        pass
    try:
        filterwheel.__del__()
    except (ValueError, TypeError, AttributeError, RuntimeError):
        pass
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
    if n_intervals and 'background' not in style:
        if not filterwheel.is_dummy:
            style['background'] = 'green'
            return style, 'Real'
    return dash.no_update


@app.callback(
    [Output('tau2-status', 'style'),
     Output('tau2-status', 'children')],
    Input('interval-component', 'n_intervals'),
    State('tau2-status', 'style'))
def check_valid_tau(n_intervals, style):
    if n_intervals and 'background' not in style:
        if camera.is_camera_alive:
            style['background'] = 'green'
            return style, 'Real'
    return dash.no_update


@app.callback(Output('clock', 'children'), Input('interval-component', 'n_intervals'))
def clock_label_update(n_intervals):
    return datetime.now().strftime("Date %Y-%m-%d\tTime %H:%M:%S")


@app.callback(Output('counter-images', 'children'), Input('interval-component', 'n_intervals'))
def counter_label_update(n_intervals):
    return counter_images.value


@app.callback(Output('after-photo-sync-label', 'n_clicks'),
              Input('take-photo-button', 'disabled'),
              [State('save-image-checkbox', 'value'),
               State("image-sequence-length", 'value'),
               State("image-number", 'value')])
def images_handler_callback(button_state, to_save: str, length_sequence: int, num_of_images: int):
    if button_state:
        global event_finished_image
        event_finished_image.clear()

        # take images for different filters
        logger.info(f"Taking a sequence of {length_sequence} filters.")
        t_fpa_dict, t_housing_dict, images_dict, position = {}, {}, {}, 0
        for position in range(1, length_sequence + 1):
            filterwheel.position = position
            name = int(filterwheel.position.get('name', 0))
            for _ in range(num_of_images):
                image = camera.image
                if image is not None:
                    t_fpa_dict.setdefault(name, []).append(camera.fpa)
                    t_housing_dict.setdefault(name, []).append(camera.housing)
                    images_dict.setdefault(name, []).append(image)
        for name in images_dict.keys():
            t_fpa_dict[name] = float(np.mean(t_fpa_dict[name]))
            t_housing_dict[name] = float(np.mean(t_housing_dict[name]))
            images_dict[name] = np.stack(images_dict[name])
        logger.info(f"Taken an image in position {position}.")

        if to_save:
            keys = list(images_dict.keys())
            time_current = datetime.now().strftime("%Y%m%d_h%Hm%Ms%S")
            path = (SAVE_PATH / f'cnt{counter_images.value}_{time_current}').with_suffix('.npy')
            np.save(file=path, arr=np.stack([images_dict[k] for k in keys]))
            df = pd.DataFrame(columns=[const.T_FPA, const.T_HOUSING, 'Wavelength'],
                              data=np.stack([(t_fpa_dict[k] / 100, t_housing_dict[k] / 100, k) for k in keys]))
            df['Wavelength'] = df['Wavelength'].astype('int')
            df.to_csv(path_or_buf=path.with_suffix('.csv'))
            counter_images.value = counter_images.value + 1
        logger.info("Taken an image." if 'save' not in to_save else f"Saved image number {counter_images.value}.")

        global image_storage
        image_storage = {k: v.mean(0) for k, v in images_dict.items()}

        event_finished_image.set()
        return 1
    return dash.no_update
