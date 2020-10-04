import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from server.layouts import main_layout
from server.app import app, PATHNAME_MAIN, PATHNAME_INIT_DEVICES
import server.callbacks

tab_style = {'border': '1px solid black'}
app.layout = html.Div([dcc.Location(id='url', refresh=True),
                       html.Table([html.Tr([
                           html.Td([dcc.Link('Go main', href=PATHNAME_MAIN)], style=tab_style),
                           html.Td([dcc.Link('Go to Init', href=PATHNAME_INIT_DEVICES)], style=tab_style),
                           html.Td([dcc.Link('Go to App 2', href='/apps/app2')], style=tab_style)])],
                           style=tab_style),
                       html.Div(id='page-content')])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == PATHNAME_MAIN:
        return main_layout
    elif pathname == PATHNAME_INIT_DEVICES:
        return init_layout
    elif pathname == '/apps/app2':
        return layout2
    else:
        return '404'


if __name__ == '__main__':
    PORT = 8000
    # IP = "127.0.0.1"
    IP = "0.0.0.0"
    print(f"http://{IP:s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=True, host=IP, port=PORT, threaded=True)

# todo: AlliedVisionCamera should take images in all cameras simultaneity

# todo: get_alliedvision_grabber() should actually call the multiframegrabber

# todo: multiple cameras - should try to run as many cameras as possible. When detecting a camera, her page becomes available.
## "take a photo" takes a photo in all cameras at the same time.

# todo: seperate FilterWheel and AlliedVisionCtrl. Create another class to combine them.

# todo: add ability to change f_number, focal_length to AlliedVision

# todo: change README file parsing section
