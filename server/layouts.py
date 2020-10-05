import dash_core_components as dcc
import dash_html_components as html
from utils.constants import DEFAULT_FILTER_NAMES_DICT
from server.server_utils import make_devices_names_radioitems, make_camera_models_dropdown_options_list

FONT_SIZE = {'font-size': '16px'}
OPTICS_STYLE_DICT = dict(min=0.0, type='number', style=FONT_SIZE)
main_layout = [
    html.Div([html.Div(id='use-real-filterwheel-midstep', hidden=True, children=None),
              dcc.Checklist(id='use-real-filterwheel',
                            options=[{'label': 'Real FilterWheel', 'value': 'real_filterwheel'}],
                            value=['real_filterwheel'],
                            labelStyle={'font-size': '24px', 'display': 'block'})]),
    html.Hr(),

    make_devices_names_radioitems(),
    html.Hr(),

    html.Div(children='Camera Model'),
    dcc.Dropdown(id='camera-model-dropdown', clearable=False, options=[]),
    html.Hr(),

    html.Table([
        html.Tr([html.Td([html.Div(id='focal-length-label', children='Focal Length [mm]'),
                          dcc.Input(id="focal-length",placeholder="Focal Length",
                                    value=0, **OPTICS_STYLE_DICT)]),
                 html.Td([html.Div(children='F#'),
                          dcc.Input(id="f-number", placeholder="F#",
                                    value=0, **OPTICS_STYLE_DICT)])])]),
    html.Hr(),

    html.Div(id='exposure-label', children='Exposure Time [micro sec]'),
    dcc.RadioItems(id='exposure-type-radio',value='manual',
                   options=[{'label': 'Manual', 'value': 'manual'}, {'label': 'Auto', 'value': 'auto'}]),
    dcc.Input(id="exposure-time", type='number', placeholder="Exposure Time", value=5000.0, style=FONT_SIZE),
    html.Hr(),

    html.Table([html.Tr([
        html.Td([html.Div(children='Gain [dB]'),
                 dcc.Input(id="gain", type='number', placeholder="Gain", value=0.0, style=FONT_SIZE)]),
        html.Td([html.Div(children='Gamma'),
                 dcc.Input(id="gamma", type='number', placeholder="Gamma", value=1.0, style=FONT_SIZE)])])]),
    html.Hr(),

    html.Div([html.Div("Give names to the filters:", id='filter_names_div'),
              html.Table([dcc.Input(id=f"filter_{idx}",
                                    value=DEFAULT_FILTER_NAMES_DICT[idx], style=FONT_SIZE,
                                    type='text') for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)])]),
    html.Div([html.Div("Set number of filters to be photographed:", id='image_seq_len_div'),
              dcc.Input(id='image_sequence_length', value=1, type='number', min=1, style=FONT_SIZE,
                        max=len(DEFAULT_FILTER_NAMES_DICT))]),
    html.Hr(),

    dcc.Checklist(id='save-image-checkbox', options=[{'label': 'Save Image', 'value': 'save_image'}],
                  value=[], labelStyle={'font-size': '20px', 'display': 'block'}),
    html.Hr(),
    html.Table(html.Tr(
        [html.Td(html.Button('Take a photo', id='take_photo_button', n_clicks_timestamp=0, n_clicks=0, disabled=False,
                             style={'padding': '6px 10px', 'text-align': 'center', 'font-size': '20px'})),
         html.Td(dcc.Upload(
             html.Button('Upload a photo',
                         style={'padding': '6px 10px', 'text-align': 'center', 'font-size': '20px'})
             , id='upload_img'))])),
    html.Ul(id='file_list'),
    html.Div(id='after_take_photo', hidden=True),
    html.Div(children=[], id='imgs'),
]
