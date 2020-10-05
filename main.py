import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from server.layouts import main_layout
from server.app import app, PATHNAME_MAIN, PATHNAME_INIT_DEVICES
import server.callbacks

app.layout = html.Div(id='page-content', children=main_layout)


if __name__ == '__main__':
    PORT = 8000
    # IP = "127.0.0.1"
    IP = "0.0.0.0"
    print(f"http://{IP:s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=True, host=IP, port=PORT, threaded=True)

# todo: AlliedVisionCamera should take download in all cameras simultaneity

# todo: get_alliedvision_grabber() should actually call the multiframegrabber

# todo: multiple cameras - should try to run as many cameras as possible. When detecting a camera, her page becomes available.
## "take a photo" takes a photo in all cameras at the same time.

# todo: seperate FilterWheel and AlliedVisionCtrl. Create another class to combine them.

# todo: change README file parsing section
