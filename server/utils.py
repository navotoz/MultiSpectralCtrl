import struct
import os
from pathlib import Path
import numpy as np
from PIL import Image
from base64 import b64encode, b64decode
from io import BytesIO
import dash_html_components as html
from urllib.parse import quote as urlquote
from utils.constants import SAVE_PATH, IMAGE_FORMAT, DISPLAY_IMAGE_SIZE, TIFF_NOTES, TIFF_X_RESOLUTION, TIFF_Y_RESOLUTION
from multiprocessing.dummy import Pool
import dash_core_components as dcc
from devices import valid_cameras_names_list, TIFF_MODEL_NAME
from datetime import datetime
import matplotlib.pyplot as plt
if not SAVE_PATH.is_dir():
    SAVE_PATH.mkdir()


def get_image_filename(tiff_tags: dict, filter_names_list: list) -> Path:
    filename = datetime.now().strftime('d20%y%m%d_h%Hm%Ms%S_')
    filename += f"{tiff_tags.get(TIFF_MODEL_NAME, '')}"
    if filter_names_list:
        filename += f"_{len(filter_names_list)}Filters_"
        filename += '_'.join(filter_names_list)
    filename += '.tiff'
    return Path(filename)


def get_filter_names_list(image_list: list) -> list:
    if isinstance(image_list[0], tuple):
        lst = filter(lambda x: isinstance(x, tuple), image_list)
        return list(map(lambda x: x[0], lst))
    return []


def get_filters_tags_images(image_list: list) -> tuple:
    filter_names_list = get_filter_names_list(image_list)
    tiff_tags = list(filter(lambda x: isinstance(x, dict), image_list))[-1]
    image_list = filter(lambda x: not isinstance(x, dict), image_list)
    image_list = map(lambda x: x[-1] if isinstance(x, tuple) else x, image_list)
    return filter_names_list, tiff_tags, list(image_list)


def save_image_to_tiff(image_list: list):
    filter_names_list, tiff_tags, image_list = get_filters_tags_images(image_list)
    image_list = list(map(lambda image: Image.fromarray(image.astype('uint16')), image_list))
    full_path = SAVE_PATH / get_image_filename(tiff_tags, filter_names_list)
    first_image = image_list.pop(0)
    h,w = first_image.size
    tiff_tags[TIFF_Y_RESOLUTION], tiff_tags[TIFF_X_RESOLUTION] = h,w
    tiff_tags[TIFF_NOTES] += f'Height{h};Width{w};'
    first_image.save(full_path, format=IMAGE_FORMAT, tiffinfo=tiff_tags,
                     append_images=image_list, save_all=True, compression=None, quality=100)


def base64_to_split_numpy_image(base64_string: str) -> list:
    text_base64  = base64_string.split('base64,')[-1]
    bytes_base64 = text_base64.encode()
    buffer = b64decode(bytes_base64)
    height, width = buffer.find(b';Height'), buffer.find(b';Width')
    height = buffer[height:height + buffer[height+1:].find(b';')+1].decode()
    height = int(''.join([x for x in height if x.isdigit()]))
    width = buffer[width:width + buffer[width+1:].find(b';')+1].decode()
    width = int(''.join([x for x in width if x.isdigit()]))
    image_numpy = np.frombuffer(buffer, dtype='uint16')
    image_numpy = image_numpy[len(image_numpy) % (height * width):]
    ch = len(image_numpy) // (height * width)
    image_numpy = image_numpy.reshape(ch, width, height)
    return list(map(lambda im: im.squeeze(), np.split(image_numpy, image_numpy.shape[0])))


def numpy_to_base64(image_: (np.ndarray, Image.Image)) -> bytes:
    if isinstance(image_, Image.Image):
        image_ = np.array(image_)
    image_ = image_.astype('float')
    image_ -= np.amin(image_)
    image_ = image_ / (np.amax(image_)+np.finfo(image_.dtype).eps)
    image_ *= 255
    image_ = image_.astype('uint8')
    image_bytes = BytesIO()
    Image.fromarray(image_).save(image_bytes, 'jpeg')
    return image_bytes.getvalue()


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


def make_image_html(input_tuple: tuple) -> html.Td:
    name, image = input_tuple
    img = f"data:image/jpeg;base64,{b64encode(numpy_to_base64(image)).decode('utf-8'):s}"
    return html.Td([html.Div(name), html.Img(src=img, style={'width': DISPLAY_IMAGE_SIZE})])


def make_images_for_web_display(image_list: list) -> html.Tr:
    with Pool(len(image_list)) as pool:
        return html.Tr(list(pool.imap(make_image_html, image_list)))


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
                               value='none',
                               labelStyle={'font-size': '20px', 'display': 'block'})], style=TAB_STYLE))
    return html.Table([html.Tr(tr_list)], style=TAB_STYLE, id='devices-radioitems-table')


def make_models_dropdown_options_list(camera_state_list: list):
    return [{'label': name, 'value': name} for name, state in camera_state_list if 'none' not in state]


def list_server_routes(server):
    routes = []
    for rule in server.url_map.iter_rules():
        routes.append('%s' % rule)
    return routes


def show_image(image: (Image.Image, np.ndarray), title=None, v_min=None, v_max=None, to_close:bool=True):
    if isinstance(image, Image.Image):
        image = np.array([image])
    if np.any(np.iscomplex(image)):
        image = np.abs(image)
    if len(image.shape) > 2:
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = image.transpose((1, 2, 0))  # CH x W x H -> W x H x CH
        elif image.shape[-1] != 3 and image.shape[-1] != 1:  # CH are not RGB or grayscale
            image = image.mean(-1)
    plt.imshow(image.squeeze(), cmap='gray', vmin=v_min, vmax=v_max)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close() if to_close else None