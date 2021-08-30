import os
from base64 import b64encode, b64decode
from datetime import datetime
from io import BytesIO
from multiprocessing.dummy import Pool
from pathlib import Path
from time import time_ns, sleep
from urllib.parse import quote as urlquote

import dash_html_components as html
import numpy as np
from PIL import Image

import utils.constants as const
from utils.constants import SAVE_PATH, IMAGE_FORMAT, DISPLAY_IMAGE_SIZE, TIFF_NOTES, TIFF_X_RESOLUTION, \
    TIFF_Y_RESOLUTION

if not SAVE_PATH.is_dir():
    SAVE_PATH.mkdir()


def get_image_filename(tiff_tags: dict, filter_names_list: list) -> Path:
    filename = datetime.now().strftime('d20%y%m%d_h%Hm%Ms%S_')
    filename += f"{tiff_tags.get(const.TIFF_MODEL_NAME, '')}"
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
    h, w = first_image.size
    tiff_tags[TIFF_Y_RESOLUTION], tiff_tags[TIFF_X_RESOLUTION] = h, w
    tiff_tags[TIFF_NOTES] += f'Height{h};Width{w};'
    first_image.save(full_path, format=IMAGE_FORMAT, tiffinfo=tiff_tags,
                     append_images=image_list, save_all=True, compression=None, quality=100)


def base64_to_split_numpy_image(base64_string: str) -> list:
    text_base64 = base64_string.split('base64,')[-1]
    bytes_base64 = text_base64.encode()
    buffer = b64decode(bytes_base64)
    height, width = buffer.find(b';Height'), buffer.find(b';Width')
    height = buffer[height:height + buffer[height + 1:].find(b';') + 1].decode()
    height = int(''.join([x for x in height if x.isdigit()]))
    width = buffer[width:width + buffer[width + 1:].find(b';') + 1].decode()
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
    image_ = image_ / (np.amax(image_) + np.finfo(image_.dtype).eps)
    image_ *= 255
    image_ = image_.astype('uint8')
    image_bytes = BytesIO()
    Image.fromarray(image_).save(image_bytes, 'jpeg')
    return image_bytes.getvalue()


def file_download_link(filename: Path):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = urlquote(str(filename))
    if isinstance(filename, Path):
        return html.A(filename.name, href=location)  # , download=True)
    return html.A(filename, href=location)  # , download=True)


def find_files_in_savepath(endswith: str) -> list:
    """List the files in the upload directory."""
    return list(SAVE_PATH.glob(f'*.{endswith}'))


def make_links_from_files(file_list: (list, tuple)) -> list:
    return [html.Li(file_download_link(filename)) for filename in file_list]


def make_image_html(input_tuple: tuple) -> html.Td:
    name, image = input_tuple
    img = f"data:image/jpeg;base64,{b64encode(numpy_to_base64(image)).decode('utf-8'):s}"
    return html.Td([html.Div(name), html.Img(src=img, style={'width': DISPLAY_IMAGE_SIZE})])


def make_images_for_web_display(image_list: list) -> html.Tr:
    with Pool(len(image_list)) as pool:
        return html.Tr(list(pool.imap(make_image_html, image_list)))


def list_server_routes(server):
    routes = []
    for rule in server.url_map.iter_rules():
        routes.append('%s' % rule)
    return routes


def decorate_all_functions(function_decorator):
    def decorator(cls):
        for name, obj in vars(cls).items():
            if callable(obj):
                setattr(cls, name, function_decorator(obj))
        return cls

    return decorator


def wait_for_time(func, wait_time_in_nsec: float = 1e9):
    def do_func(*args, **kwargs):
        start_time = time_ns()
        res = func(*args, **kwargs)
        sleep(max(0.0, 1e-9 * (start_time + wait_time_in_nsec - time_ns())))
        return res

    return do_func


def only_numerics(seq):
    if seq is None:
        return -1
    if isinstance(seq, str):
        seq_type = type(seq)
        return int(seq_type().join(filter(seq_type.isdigit, seq)))
    return seq
