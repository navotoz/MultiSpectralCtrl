from typing import Dict

from flask import Response, url_for
import dash_html_components as html
from utils.constants import DISPLAY_IMAGE_SIZE
from server.app import server, cameras_dict
from server.utils import numpy_to_base64

from threading import Thread
from collections import deque
from collections.abc import Generator
import cv2
from server.utils import show_image


class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, camera_name:(str,None)):
        self._camera_name = camera_name
        self._iterator = CameraIterator(None)
        self._queue = deque(maxlen=1)
        self._queue.append(b'')
        self._thread = None

    def _run(self):
        self._queue.append(self._iterator)

    def __call__(self):
        self._iterator._camera = cameras_dict[self._camera_name] if self._camera_name else None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        for value in self._queue.pop():
            yield value
        self._thread.join()


class CameraIterator(Generator):
    def __init__(self, camera):
        super().__init__()
        self._camera = camera
        self._frame_number = 0

    def __del__(self):
        self._camera = None

    @property
    def frame_number(self):
        self._frame_number += 1
        return self._frame_number

    def throw(self, typ, val, tb):
        raise StopIteration

    def close(self) -> None:
        self._camera = None

    def send(self, value):
        image = self._camera()
        w, h = image.shape
        res = cv2.putText(image, f"{self.frame_number}", (h-200, w-200), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 2)
        image = numpy_to_base64(res) if self._camera else b''
        return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


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
        streamers_dict.setdefault(name, ThreadedGenerator(name))
        children_list.append(html.Hr())
    return html.Div([*children_list])


streamers_dict = dict()
