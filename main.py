from server.layouts import main_layout
from server.app import app
import server.callbacks

app.layout = main_layout


if __name__ == '__main__':
    PORT = 8000
    # IP = "127.0.0.1"
    IP = "0.0.0.0"
    print(f"http://{IP:s}:{PORT:d}/")
    app.logger.disabled = True
    app.run_server(debug=False, host=IP, port=PORT, threaded=True)


# todo: displays images after taking. in a table - {multifilter} | {regular}

# todo: make viewer - maybe in a multipage setup

# todo: change README file parsing section and all other..

# todo: add documantiation
