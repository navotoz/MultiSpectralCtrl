import dash_core_components as dcc
import dash_html_components as html
from utils.constants import DEFUALT_FOCAL_LENGTH, DEFUALT_F_NUMBER, DEFAULT_FILTER_NAMES_DICT

FONT_SIZE = {'font-size': '16px'}

upload_image = html.Div(dcc.Upload(id='upload_img',accept='npy',multiple=False, children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
               style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                      'font-size': '26px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
                      'margin': '10px'}), hidden=True, id='upload_img_div')


main_layout = [html.Div([
    html.Div(children='Camera Model'), dcc.Dropdown(id='choose_camera_model',
                                                    options=[{'label': 'ALVIUM_1800U_1236',
                                                              'value': 'ALVIUM_1800U_1236'}], clearable=False,
                                                    value='ALVIUM_1800U_1236')]),
    html.Hr(),

    html.Table([
        html.Tr([html.Td([html.Div(id='focal_tag', children='Focal Length [mm]'),
                  dcc.Input(id="input_lens_focal_length_mm", min=0, type='number', placeholder="Focal Length",
                            value=DEFUALT_FOCAL_LENGTH, style=FONT_SIZE)]),
        html.Td([html.Div(children='F#'),
                 dcc.Input(id="input_lens_f_number", type='number', min=0, placeholder="F#", value=DEFUALT_F_NUMBER, style=FONT_SIZE)])])]),
    html.Hr(),

    html.Div(children='Exposure Time [micro sec]'),
             dcc.RadioItems(id='exposure_type_radio',
                            options=[{'label': 'Manual', 'value': 'manual'}, {'label': 'Auto', 'value': 'auto'}],
                            value='auto'),
             dcc.Input(id="input_exposure_time", type='number', placeholder="Exposure Time", style=FONT_SIZE),
    html.Hr(),

    html.Table([html.Tr([
        html.Td([html.Div(children='Gain [dB]'),
                 dcc.Input(id="input_gain_time", type='number', placeholder="Gain", value=0.0, style=FONT_SIZE)]),
        html.Td([html.Div(children='Gamma'),
                 dcc.Input(id="input_gamma_time", type='number', placeholder="Gamma", value=1.0, style=FONT_SIZE)])])]),
    html.Hr(),

    html.Div([html.Div("Give names to the filters:", id='filter_names_div'),
              html.Table([dcc.Input(id=f"filter_{idx}",
                                    value=DEFAULT_FILTER_NAMES_DICT[idx],  style=FONT_SIZE,
                                    type='text') for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)])]),
    html.Div([html.Div("Set number of filters to be photographed:", id='image_seq_len_div'),
              dcc.Input(id='image_sequence_length', value=1, type='number', min=1, style=FONT_SIZE,
                        max=len(DEFAULT_FILTER_NAMES_DICT))]),
    html.Hr(),

    dcc.Checklist(id='save_img_checkbox', options=[{'label': 'Save images', 'value': 'save_img'},
                                                   {'label': 'Display images', 'value': 'disp_img'}],
                  value=['disp_img'], labelStyle={'font-size': '24px', 'display': 'block'}),
    html.Hr(),
    html.Table(html.Tr([html.Td(html.Button('Take a photo', id='photo_button', n_clicks_timestamp=0, n_clicks=0, disabled=False,
                        style={'padding':'6px 10px', 'text-align':'center', 'font-size':'22px'})),
                        html.Td(dcc.Upload(
                            html.Button('Upload a photo',
                                        style={'padding':'6px 10px', 'text-align':'center', 'font-size':'22px'})
                            ,id='upload_img'))])),
    html.Ul(id='file_list'),
    html.Div(id='photo_but_div', hidden=True),
    html.Div(children=[], id='imgs'),
]
