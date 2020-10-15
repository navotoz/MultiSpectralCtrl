from flask import Response, url_for
import dash_html_components as html
from utils.constants import DISPLAY_IMAGE_SIZE
from server.app import server, cameras_dict
from server.utils import numpy_to_base64

from threading import Thread
from collections import deque
from collections.abc import Generator


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
        self._iterator.camera = cameras_dict[self._camera_name] if self._camera_name else None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        for value in self._queue.pop():
            yield value
        self._thread.join()


class CameraIterator(Generator):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def __del__(self):
        self.camera = None

    def throw(self, typ, val, tb):
        raise StopIteration

    def close(self) -> None:
        self.camera = None

    def send(self, value):
        image = numpy_to_base64(self.camera()) if self.camera else b''
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
