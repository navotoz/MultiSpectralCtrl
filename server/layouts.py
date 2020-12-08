import dash_core_components as dcc
import dash_html_components as html
from devices.FilterWheel import DEFAULT_FILTER_NAMES_DICT
from server.utils import make_devices_names_radioitems
from utils.constants import UPDATE_INTERVAL_SECONDS

FONT_SIZE = {'font-size': '16px'}
OPTICS_STYLE_DICT = dict(type='text', style=FONT_SIZE, debounce=True)
main_layout = [
    html.Table(html.Tr(
        [html.Td(dcc.Link('Camera Viewer', id='viewer-link', href='/viewer', target='_blank')),
         html.Td(html.Button('Kill',
                             style={'padding': '1px 4px', 'text-align': 'center', 'font-size': '20px'}),
                 id='kill-button')])),
    html.Hr(),

    dcc.Checklist(id='use-real-filterwheel', options=[{'label': 'Real FilterWheel', 'value': 'real'}],
                  value=['real'], labelStyle={'font-size': '24px', 'display': 'block'}),
    html.Hr(),

    make_devices_names_radioitems(),
    html.Hr(),

    html.Div(children='Camera Model'),
    dcc.Dropdown(id='camera-model-dropdown', clearable=False, options=[]),
    html.Hr(),

    html.Div(children='Which camera is connected to the FilterWheel'),
    dcc.RadioItems(id='multispectral-camera-radioitems', options=[]),
    html.Hr(),

    html.Table([
        html.Tr([html.Td([html.Div(id='focal-length-label', children='Focal Length [mm]'),
                          dcc.Input(id="focal-length", placeholder="Focal Length", value=0, **OPTICS_STYLE_DICT)]),
                 html.Td([html.Div(children='F#'),
                          dcc.Input(id="f-number", placeholder="F#", value=0, **OPTICS_STYLE_DICT)])])]),
    html.Hr(),

    html.Div(id='exposure-label', children='Exposure Time [micro sec]'),
    dcc.RadioItems(id='exposure-type-radio', value='manual',
                   options=[{'label': 'Manual', 'value': 'manual'}, {'label': 'Auto', 'value': 'auto'}]),
    dcc.Input(id="exposure-time", placeholder="Exposure Time", value=5000,  **OPTICS_STYLE_DICT),
    html.Hr(),

    html.Table([html.Tr([
        html.Td([html.Div(children='Gain [dB]'),
                 dcc.Input(id="gain", placeholder="Gain", value=0, **OPTICS_STYLE_DICT)]),
        html.Td([html.Div(children='Gamma'),
                 dcc.Input(id="gamma", placeholder="Gamma", value=1,  **OPTICS_STYLE_DICT)])])]),
    html.Hr(),

    html.Div([html.Div("Give names to the filters. 0 is for glass:", id='filter-names-label'),
              html.Table([dcc.Input(id=f"filter-{idx}", **OPTICS_STYLE_DICT,value=DEFAULT_FILTER_NAMES_DICT[idx])
                          for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)])]),
    html.Div([html.Div("Set number of filters to be photographed:", id='image-sequence-length-label'),
              dcc.Input(id='image-sequence-length', value=1, type='number', min=1, style=FONT_SIZE, debounce=True,
                        max=len(DEFAULT_FILTER_NAMES_DICT))]),
    html.Hr(),

    dcc.Checklist(id='save-image-checkbox', options=[{'label': 'Save Image', 'value': 'save'}],
                  value=[], labelStyle={'font-size': '20px', 'display': 'block'}),
    html.Hr(),

    html.Table(html.Tr(
        [html.Td(html.Button('Take a photo', id='take-photo-button', n_clicks_timestamp=0, n_clicks=0, disabled=False,
                             style={'padding': '6px 10px', 'text-align': 'center', 'font-size': '20px'})),
         html.Td(dcc.Upload(html.Button('Upload a photo',
                                        style={'padding': '6px 10px', 'text-align': 'center', 'font-size': '20px'}),
                            id='upload-img-button'))])),
    html.Ul(id='file-list'),
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_SECONDS * 1e3),  # in milliseconds
    html.Div(id='after-photo-sync-label', hidden=True),
    html.Div(children=html.Table(html.Tr(id='imgs', children=[]))),
    html.Hr(),

    html.Div(id='log-div', children=[])

    ]
    # html.Div(id='log-div',style=dict(height='200px', overflow='auto'))]
