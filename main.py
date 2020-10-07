import dash
from server.layouts import main_layout
from server.app import app, cameras_dict
import server.callbacks
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from server.viewer import make_viewers

app.layout = html.Div([dcc.Location(id='url', refresh=False), html.Div(id='page-content')])
layout = main_layout


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'),
              State('page-content', 'children'))
def display_page(pathname, current_layout):
    global layout
    global cameras_dict
    if pathname == '/':
        for name in cameras_dict.keys():
            try:
                cameras_dict[name].flag_stream = False
            except (AttributeError, KeyError):
                pass
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


# todo: add more types of cameras

# todo: fix uploads

# todo: change README file parsing section and all other..

# todo: add documantiation
