# from flask import Response, url_for
# import dash_html_components as html
# import dash_core_components as dcc
# from utils.constants import DISPLAY_IMAGE_SIZE
# from server.app import server, cameras_dict
# from server.utils import numpy_to_base64
# import numpy as np
# from devices import valid_cameras_names_list



from threading import Thread, Semaphore
from collections import deque
class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator, sentinel=object(), queue_maxsize=1, daemon=False, Thread=Thread, Queue=deque):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxlen=queue_maxsize)
        self._queue.append(b'')
        self._thread = Thread(name=repr(iterator), target=self._run      )
        self._thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            self._queue.append(self._iterator())
        finally:
            self._queue.append(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.pop(), self._sentinel):
            yield value

        self._thread.join()



class Streamer:
    def __init__(self):
        super().__init__()
        self.flag_stream = False
        self.camera = None
        self.__semaphore = Semaphore(1)

    def __del__(self):
        self.flag_stream = False
        self.__semaphore.acquire()
        self.camera = None


    def __call__(self):
        print('call')
        while True:
            if self.camera and self.flag_stream:
                while self.camera and self.flag_stream:
                    self.__semaphore.acquire()
                    try:
                        # image = numpy_to_base64(self.camera())
                        image = b'a'
                        self.__semaphore.release()
                    except:
                        self.__semaphore.release()
                        break
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
            else:
                yield b''


[x for x in ThreadedGenerator(Streamer())]
#
#
#
# @server.route("/video_feed/<name>")
# def video_feed(name):
#     return Response(streamers_dict[name](), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# def make_viewers() -> html.Div:
#     global cameras_dict
#     dict_available_cameras = list(filter(lambda item: item[-1], cameras_dict.items()))
#     children_list = []
#     for name, camera in dict_available_cameras:
#         children_list.append(html.Div(name))
#         children_list.append(html.Img(src=url_for(f'video_feed', name=name), style={'width': DISPLAY_IMAGE_SIZE}))
#         streamers_dict.setdefault(name, Streamer()).camera = cameras_dict[name]
#         streamers_dict.setdefault(name, Streamer()).flag_stream = True
#         children_list.append(html.Hr())
#     return html.Div([*children_list])
#
#
# streamers_dict = dict().fromkeys(valid_cameras_names_list, Streamer())
# [print(val.is_alive() for val in streamers_dict.values()]
# # [val.start() for val in streamers_dict.values()]
