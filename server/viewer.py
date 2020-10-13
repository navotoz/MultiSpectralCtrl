from flask import Response, url_for
import dash_html_components as html
import dash_core_components as dcc
from utils.constants import DISPLAY_IMAGE_SIZE
from server.app import server, cameras_dict
from server.utils import numpy_to_base64
import numpy as np


class Streamer:
    def __init__(self):
        self.flag_stream = False
        self.camera = None

    def __call__(self):
        while True:
            if self.camera and self.flag_stream:
                while self.camera and self.flag_stream:
                    try:
                        image = numpy_to_base64(self.camera())
                    except:
                        break
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
            else:
                yield b''


@server.route("/video_feed/<name>")
def video_feed(name):
    return Response(streamers_dict[name](), mimetype='multipart/x-mixed-replace; boundary=frame')


def make_viewers() -> html.Div:
    global cameras_dict
    dict_available_cameras = list(filter(lambda item: item[-1], cameras_dict.items()))
    children_list = []
    for name, camera in dict_available_cameras:
        children_list.append(html.Div(name))
        children_list.append(html.Img(src=url_for(f'video_feed', name=name), style={'width': DISPLAY_IMAGE_SIZE}))
        streamers_dict.setdefault(name, Streamer()).camera = cameras_dict[name]
        streamers_dict.setdefault(name, Streamer()).flag_stream = True
        children_list.append(html.Hr())
    return html.Div([dcc.Link('Control Page', href='/'),
                     html.Hr(),
                     *children_list])


streamers_dict = dict()
