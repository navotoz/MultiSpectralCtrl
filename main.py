from signal import SIGINT, signal, SIGTERM
from socket import gethostname

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from server.app import app
from server.layouts import main_layout
from server.viewer import make_viewers, streamers_dict

app.layout = html.Div([dcc.Location(id='url', refresh=False),
                       html.Div(id='page-content', children=main_layout)])
layout = main_layout
import server.callbacks


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'),
              State('page-content', 'children'))
def display_page(pathname, current_layout):
    global layout
    if pathname == '/':
        for name in streamers_dict.keys():
            streamers_dict[name].flag_stream = False
            streamers_dict[name]._camera = None
        return layout
    elif pathname == '/viewer':
        layout = current_layout
        return make_viewers()
    else:
        return dash.no_update


if __name__ == '__main__':
    signal(SIGINT, server.callbacks.exit_handler)
    signal(SIGTERM, server.callbacks.exit_handler)
    PORT = 8080
    IP = "0.0.0.0"
    # noinspection HttpUrlsUsage
    print(f"http://{gethostname():s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=False, host=IP, port=PORT, threaded=True)
