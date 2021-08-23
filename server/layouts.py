import dash_core_components as dcc
import dash_html_components as html

from devices.FilterWheel import DEFAULT_FILTER_NAMES_DICT
from utils.constants import UPDATE_INTERVAL_SECONDS

FONT_SIZE = {'font-size': '16px'}
OPTICS_STYLE_DICT = dict(type='text', style=FONT_SIZE, debounce=True)
    # html.Div(id='log-div',style=dict(height='200px', overflow='auto'))]
main_layout = [
    html.Table(html.Tr(
        [html.Td(dcc.Link('Camera Viewer', id='viewer-link', href='/viewer', target='_blank')),
         html.Td(html.Button('Kill',
                             style={'padding': '1px 4px', 'text-align': 'center', 'font-size': '20px'}),
                 id='kill-button')])),
    html.Hr(),

    html.Table(
        html.Tr([
            html.Td(html.Div('Filterwheel', id='filterwheel-status-label',
                             style={'padding': '6px 10px', 'border': None, 'text-align': 'center', 'font-size': '20px'})),
            html.Td(html.Div('Dummy', id='filterwheel-status',
                             style={'padding': '6px 10px', 'border': '1px solid black',
                                    'text-align': 'center', 'font-size': '20px'}))
         ])
    ),
    html.Table(
        html.Tr([
            html.Td(html.Div('Tau2', id='tau2-status-label',
                             style={'padding': '6px 10px', 'border': None, 'text-align': 'center',
                                    'font-size': '20px'})),
            html.Td(html.Div('Dummy', id='tau2-status',
                             style={'padding': '6px 10px', 'border': '1px solid black',
                                    'text-align': 'center', 'font-size': '20px'}))
        ])
    ),
    html.Hr(),

    html.Div([html.Div("Give names to the filters. 0 is for glass:", id='filter-names-label'),
              html.Table([dcc.Input(id=f"filter-{idx}", **OPTICS_STYLE_DICT,value=DEFAULT_FILTER_NAMES_DICT[idx])
                          for idx in range(1, len(DEFAULT_FILTER_NAMES_DICT) + 1)])]),
    html.Div([html.Div("Set number of filters to be photographed:", id='image-sequence-length-label'),
              dcc.Input(id='image-sequence-length', value=6, type='number', min=1, style=FONT_SIZE, debounce=True,
                        max=len(DEFAULT_FILTER_NAMES_DICT))]),
    html.Hr(),

    dcc.Checklist(id='save-image-checkbox', options=[{'label': 'Save Image', 'value': 'save'}],
                  value=['save'], labelStyle={'font-size': '20px', 'display': 'block'}),
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
