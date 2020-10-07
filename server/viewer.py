from flask import Response, url_for
import dash_html_components as html
import dash_core_components as dcc
from utils.constants import DISPLAY_IMAGE_SIZE
from server.app import server, cameras_dict


@server.route("/video_feed/<name>")
def video_feed(name):
    return Response(cameras_dict[name].stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


def make_viewers() -> html.Div:
    global cameras_dict
    dict_available_cameras = list(filter(lambda item: item[-1], cameras_dict.items()))
    children_list = []
    for name, camera in dict_available_cameras:
        children_list.append(html.Div(name))
        children_list.append(html.Img(src=url_for(f'video_feed', name=name), style={'width': DISPLAY_IMAGE_SIZE}))
        cameras_dict[name].flag_stream=True
        children_list.append(html.Hr())
    return html.Div([dcc.Link('Control Page', href='/'),
                    html.Hr(),
                    *children_list])
