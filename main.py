from server.layouts import main_layout
from server.app import app
import server.callbacks

app.layout = main_layout


if __name__ == '__main__':
    PORT = 8000
    IP = "0.0.0.0"
    print(f"http://{IP:s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=False, host=IP, port=PORT, threaded=True)


# todo: make viewer - maybe in a multipage setup
# todo: add stream() to the camera, so no need to do with... each time. Use yield to avoid __exit__

# todo: add more types of cameras

# todo: change README file parsing section and all other..

# todo: add documantiation
