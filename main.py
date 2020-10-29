import dash
from server.layouts import main_layout
from server.app import app, cameras_dict
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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
    global cameras_dict
    if pathname == '/':
        for name in streamers_dict.keys():
            streamers_dict[name].flag_stream = False
            streamers_dict[name].camera = None
        return layout
    elif pathname == '/viewer':
        layout = current_layout
        return make_viewers()
    else:
        return dash.no_update


if __name__ == '__main__':
    PORT = 8000
    IP = "0.0.0.0"
    print(f"http://{IP:s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=False, host=IP, port=PORT, threaded=True)

# todo: change README file parsing section and all other..

# todo: add documantiation
