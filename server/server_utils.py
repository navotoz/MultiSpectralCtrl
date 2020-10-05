import os
from pathlib import Path
import numpy as np
from PIL import Image
from base64 import b64encode, b64decode
from io import BytesIO
import dash_html_components as html
from urllib.parse import quote as urlquote
from utils.constants import INIT_EXPOSURE, SAVE_PATH, IMAGE_FORMAT
from multiprocessing.dummy import Pool
import dash_core_components as dcc
from devices.CamerasCtrl import valid_cameras_names_list

if not SAVE_PATH.is_dir():
    SAVE_PATH.mkdir()


def save_image(image_grabber, to_save_img: bool) -> dict:
    multi_frame_images_dict, image_tags, f_name = image_grabber()
    if to_save_img:
        full_path = SAVE_PATH / Path(f_name)
        frames_keys_list = list(multi_frame_images_dict.keys())
        frames_keys_list.sort()
        first_key = frames_keys_list.pop()
        multi_frame_images_dict[first_key] \
            .save(full_path.with_suffix('.tiff'), format="tiff", tiffinfo=image_tags,
                  append_images=list(map(lambda key: multi_frame_images_dict[key], frames_keys_list)),
                  save_all=True, compression=None, quality=100)
    return dict(map(lambda im: (im[0], np.array(im[1])), multi_frame_images_dict.items()))


def base64_to_split_numpy_image(base64_string: str, height: int, width: int) -> list:
    buffer = b64decode(base64_string.split('base64,')[-1])
    image_numpy = np.frombuffer(buffer, dtype='uint16')
    image_numpy = image_numpy[len(image_numpy) % (height * width):]
    ch = len(image_numpy) // (height * width)
    image_numpy = image_numpy.reshape(ch, width, height)
    return list(map(lambda im: im.squeeze(), np.split(image_numpy, image_numpy.shape[0])))


def numpy_to_base64(image: np.ndarray) -> str:
    image_ = image.astype('float32').copy()
    image_ -= np.amin(image_)
    image_ = image_ / np.amax(image_)
    image_ *= 255
    image_ = image_.astype('uint8')
    image_bytes = BytesIO()
    Image.fromarray(image_).save(image_bytes, 'PNG')
    # image_bytes = cv2.imencode('.png', x)[1].tobytes()  method with cv2  # todo: check if method works on real download
    return f"data:image/png;base64,{b64encode(image_bytes.getvalue()).decode('utf-8'):s}"


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)  # , download=True)


def find_files_in_savepath(endswith: str = IMAGE_FORMAT) -> list:
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(SAVE_PATH):
        if filename.endswith(endswith):
            path = os.path.join(SAVE_PATH, filename)
            if os.path.isfile(path):
                files.append(filename)
    return files


def make_links_from_files(file_list: (list, tuple)) -> list:
    return [html.Li(file_download_link(filename)) for filename in file_list]


def make_values_dict(camera_feat_dict: dict, model_name: str) -> list:
    init_exposure = INIT_EXPOSURE // camera_feat_dict[model_name]['exposure_increment']
    init_exposure *= camera_feat_dict[model_name]['exposure_increment']
    return [camera_feat_dict[model_name]['exposure_min'] + init_exposure,
            camera_feat_dict[model_name]['exposure_min'],
            camera_feat_dict[model_name]['exposure_max'],
            camera_feat_dict[model_name]['exposure_increment'],
            camera_feat_dict[model_name]['gain_min'],
            camera_feat_dict[model_name]['gain_max'],
            camera_feat_dict[model_name]['gain_increment'],
            camera_feat_dict[model_name]['gamma_min'],
            camera_feat_dict[model_name]['gamma_max'],
            camera_feat_dict[model_name]['gamma_increment']]


def make_image_html(input_tuple: tuple) -> html.Div:
    name, img = input_tuple
    return html.Div([html.Div(name), html.Img(src=numpy_to_base64(img), style={'width': '20%'})])


def make_images(images: dict) -> html.Div:
    """
    Creates an image inside a Div in html.
    :param images: list of download as np.ndarrays.
    :return: html.Div containing the download.
    """
    if not images:
        return html.Div()
    with Pool(6) as pool:
        return html.Div(list(pool.imap(make_image_html, images.items())))


def make_devices_names_radioitems():
    TAB_STYLE = {'border': '1px solid black'}
    tr_list = []
    for name in valid_cameras_names_list:
        tr_list.append(
            html.Td([
                html.Div(id=f'{name}-type-radioboxes-label', children=f'{name}'),
                dcc.RadioItems(id=f'{name}-camera-type-radio',
                               options=[{'label': 'Real', 'value': 'real'},
                                        {'label': 'Dummy', 'value': 'dummy'},
                                        {'label': 'None', 'value': 'none'}],
                               value='dummy',
                               labelStyle={'font-size': '20px', 'display': 'block'})], style=TAB_STYLE))
    return html.Table([html.Tr(tr_list)], style=TAB_STYLE, id='devices-radioitems-table')


def make_camera_models_dropdown_options_list(camera_state_list: list):
    return [{'label': name, 'value': name} for name, state in camera_state_list if 'none' not in state]
