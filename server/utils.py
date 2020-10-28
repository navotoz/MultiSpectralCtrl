import os
from pathlib import Path
import numpy as np
from PIL import Image
from base64 import b64encode, b64decode
from io import BytesIO
import dash_html_components as html
from urllib.parse import quote as urlquote
from utils.constants import SAVE_PATH, IMAGE_FORMAT, DISPLAY_IMAGE_SIZE
from multiprocessing.dummy import Pool
import dash_core_components as dcc
from devices import valid_cameras_names_list, TIFF_MODEL_NAME
from datetime import datetime

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
    image_list = list(map(lambda image: Image.fromarray(image), image_list))
    full_path = SAVE_PATH / get_image_filename(tiff_tags, filter_names_list)
    first_image = image_list.pop(0)
    first_image.save(full_path, format=IMAGE_FORMAT, tiffinfo=tiff_tags,
                     append_images=image_list, save_all=True, compression=None, quality=100)


def base64_to_split_numpy_image(base64_string: str, n_channels: int) -> list:
    text_base64  = base64_string.split('base64,')[-1]
    bytes_base64 = text_base64.encode()
    buffer = b64decode(bytes_base64)
    Image.frombytes(mode='I',data=buffer, size=10)
    # image_numpy = np.frombuffer(buffer, dtype='uint8')
    # bit16 = 0x3FFF & image_numpy.view('uint16')
    # bit16 = bit16.reshape(n_channels, -1)
    # bit32 = 0x3FFF & image_numpy.view('uint32')
    # bit32 = bit32.reshape(n_channels, -1)
    import struct
    h_big = np.array(struct.unpack('>'+'H'*(len(buffer)//2), buffer))
    h_little = np.array(struct.unpack('<'+'H'*(len(buffer)//2), buffer))
    i_big = np.array(struct.unpack('>'+'I'*(len(buffer)//4), buffer))
    i_little = np.array(struct.unpack('<'+'I'*(len(buffer)//4), buffer))
    l_big = np.array(struct.unpack('>'+'L'*(len(buffer)//4), buffer))
    l_little = np.array(struct.unpack('<'+'L'*(len(buffer)//4), buffer))

    l = tuple(l_big[76:])
    l_big &= 0x3F
    image_numpy = image_numpy[len(image_numpy) % (height * width):]
    ch = len(image_numpy) // (height * width)
    image_numpy = image_numpy.reshape(ch, width, height)
    return list(map(lambda im: im.squeeze(), np.split(image_numpy, image_numpy.shape[0])))


def numpy_to_base64(image_: (np.ndarray, Image.Image)) -> bytes:
    if isinstance(image_, Image.Image):
        image_ = np.array(image_)
    image_ -= np.amin(image_)
    image_ = image_ / np.amax(image_)
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
