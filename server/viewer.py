from flask import Response, url_for
import dash_html_components as html
from utils.constants import DISPLAY_IMAGE_SIZE
from server.app import server, camera
from server.tools import numpy_to_base64, wait_for_time

from threading import Thread
from collections import deque
from collections.abc import Generator
import cv2
import numpy as np
from server.tools import show_image


class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, camera_name: (str, None)):
        self._camera_name = camera_name
        self._iterator = CameraIterator(None)
        self._queue = deque(maxlen=1)
        self._queue.append(b'')
        self._thread = None

    def _run(self):
        self._queue.append(self._iterator)

    def __call__(self):
        self._iterator.camera = camera if self._camera_name else None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        for value in self._queue.pop():
            yield value
        self._thread.join()


class CameraIterator(Generator):
    _get_image = None

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._frame_number = 0

    @property
    def camera(self):
        return self.__camera

    @camera.setter
    def camera(self, cam) -> None:
        self.__camera = cam
        self._get_image = wait_for_time(self.__camera, wait_time_in_nsec=1e8)

    def __del__(self):
        self.__camera = None
        self._get_image = None

    @property
    def frame_number(self):
        self._frame_number += 1
        return self._frame_number

    def throw(self, typ, val, tb):
        raise StopIteration

    def close(self) -> None:
        self.__camera = None
        self._get_image = None

    def send(self, value):
        image = self._get_image()
        w, h = image.shape
        image_upper_bound = np.iinfo(image.dtype).max
        c = int(min(image_upper_bound, image.max() + 1) if image.mean() < (image_upper_bound // 2) else 0)
        res = cv2.putText(image, f"{self.frame_number}", (h - int(0.15 * h), w - int(0.15 * w)), cv2.FONT_HERSHEY_SIMPLEX, 3, c, 2)
        image = numpy_to_base64(res) if self.camera else b''
        return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image + b'\r\n'


@server.route("/video_feed/<name>")
def video_feed(name):
    return Response(streamers_dict[name](), mimetype='multipart/x-mixed-replace; boundary=frame')


def make_viewers() -> html.Div:
    name = 'Tau2'
    children_list = []
    children_list.append(html.Div(name))
    children_list.append(html.Img(src=url_for(f'video_feed', name=name), style={'width': DISPLAY_IMAGE_SIZE}))
    streamers_dict.setdefault(name, ThreadedGenerator(name))
    children_list.append(html.Hr())
    return html.Div([*children_list])


streamers_dict = dict()
