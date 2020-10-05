import dash
from dash.dependencies import Input, Output, State
from server.app import app, handlers, filterwheel
from devices import initialize_device


@app.callback(Output('use-real-filterwheel-midstep', 'children'),
              Input('use-real-filterwheel', 'value'),
              State('use-real-filterwheel-midstep', 'children'))
def get_real_filterwheel_midstep(value, next_value):
    if value and isinstance(value, list) or isinstance(value, tuple):
        value = value[0]
    if value == next_value:
        return dash.no_update
    return value


@app.callback([Output('use-real-filterwheel', 'value'), ],
              Input('use-real-filterwheel-midstep', 'children'))
def get_real_filterwheel(value: str):
    global filterwheel
    if not value:  # use the dummy
        if not filterwheel.is_dummy:
            filterwheel = initialize_device('FilterWheel', handlers, use_dummy=True)
        return (),
    else:  # use the real FilterWheel
        try:
            filterwheel = initialize_device('FilterWheel', handlers, use_dummy=False)
        except RuntimeError:
            filterwheel = initialize_device('FilterWheel', handlers, use_dummy=True)
            return [],
        return [value],


@app.callback(Output('filter-names-label', 'n_clicks'),
              [Input('image-sequence-length-label', 'n_clicks')]+
              [Input(f"filter-{idx}", 'n_submit') for idx in range(1, filterwheel.position_count + 1)] +
              [Input(f"filter-{idx}", 'n_blur') for idx in range(1, filterwheel.position_count + 1)],
              [State(f"filter-{idx}", 'value') for idx in range(1, filterwheel.position_count + 1)])
def change_filter_names(*args):
    global filterwheel
    position_names_dict = dict(zip(range(1, filterwheel.position_count + 1), args[-filterwheel.position_count:]))
    if filterwheel.position_names_dict != position_names_dict:
        filterwheel.position_names_dict = position_names_dict
    return 1