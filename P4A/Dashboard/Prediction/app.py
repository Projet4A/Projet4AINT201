# ====================================================
# Library
# ====================================================

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc
import io
import base64
import dash_table
import pathlib
import pickle
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Initialise the app
app = dash.Dash(__name__,title='P4A INSA')
app.config.suppress_callback_exceptions = True

# ====================================================
# Path
# ====================================================

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()
MODEL_PATH = BASE_PATH.joinpath("models").resolve()

# ====================================================
# Data
# ====================================================

# Load data
parse_dates = ['DateTime_x', 'Date', 'AnalysisDate','SurveyDate']

# Read data
df = pd.read_csv(DATA_PATH.joinpath("ClientA_Data_NC.csv"), parse_dates=parse_dates)
df.RPM.fillna(df.RPM.mean(),inplace=True)

df.EquipmentClassification = df.EquipmentClassification.apply(lambda x: x.strip())

eq_class = df["EquipmentClassification"].unique()
des_area = df["Description_Area"].unique()
des_eq = df["Description_Equipment"].unique()
mpt_id = df["MptID"].unique()
des_mpt = df["Description_MeasPt"].unique()
des_apset = df["Description_APSet"].unique()
des_aparm = df["Description_AnalParm"].unique()

# load
pred = pickle.load(open(DATA_PATH.joinpath("pred.pkl"), "rb"))
real = pickle.load(open(DATA_PATH.joinpath("label.pkl"), "rb"))

# ====================================================
# Model
# ====================================================

# load
XGBC = pickle.load(open(MODEL_PATH.joinpath("xgb_clb_fsall.pkl"), "rb"))

confusion_mat = confusion_matrix(real, pred)

# ====================================================
# Function
# ====================================================

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Client A - PREDICTIONS"),
            html.H3("Select features")
        ],
    )

def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P('''Select RPM'''),
            html.Div(dcc.Input(id="dtrue",
                               type="number",
                               debounce=True,
                               placeholder="Debounce True",),
                    ),
            html.Br(),
            html.P('''Select DataValue'''),
            html.Div(dcc.Input(id="datavalue",
                               type="number",
                               debounce=True,
                               placeholder="Debounce True",),
                    ),
            html.Br(),
            html.P('''Select Units'''),
            html.Div(dcc.RadioItems(id="radio-select",
                                    options=[{'label': '0', 'value': '0'},
                                             {'label': '1', 'value': '1'},
                                             {'label': '3', 'value': '3'},
                                             {'label': '4', 'value': '4'}],
                           value='1',
                           labelStyle={'display': 'inline-block'}),
                    ),
            html.Br(),
            html.P('''Select Equipment Classification'''),
            html.Div(dcc.Dropdown(
                        id="eqclass-select",
                        options=[{"label": i, "value": i} for i in eq_class],
                        value=eq_class[0],
                                ),
                    ),
            html.Br(),
            html.P('''Select Description Area'''),
            html.Div(dcc.Dropdown(
                        id="desarea-select",
                        options=[{"label": i, "value": i} for i in des_area],
                        value=des_area[0],
                                 ),
                    ),
            html.Br(),
            html.P('''Select Description Equipment'''),
            html.Div(dcc.Dropdown(
                        id="deseq-select",
                        options=[{"label": i, "value": i} for i in des_eq],
                        value=des_eq[0],
                                 ),
                    ),
            html.Br(),
            html.P('''Select MptID'''),
            html.Div(dcc.Dropdown(
                        id="mptid-select",
                        options=[{"label": i, "value": i} for i in mpt_id],
                        value=mpt_id[0],
                                 ),
                    ),
            html.Br(),
            html.P('''Select Description MeasPt'''),
            html.Div(dcc.Dropdown(
                        id="desmpt-select",
                        options=[{"label": i, "value": i} for i in des_mpt],
                        value=des_mpt[0],
                                 ),
                    ),
            html.Br(),
            html.P('''Select Description APSet'''),
            html.Div(dcc.Dropdown(
                        id="desapset-select",
                        options=[{"label": i, "value": i} for i in des_apset],
                        value=des_apset[0],
                                ),
                    ),
            html.Br(),
            html.P('''Select Description AnalParm'''),
            html.Div(dcc.Dropdown(
                        id="desaparm-select",
                        options=[{"label": i, "value": i} for i in des_aparm],
                        value=des_aparm[0],
                                ),
                    ),
            html.Br(),
            html.Div(
                id="predict-btn-outer",
                children=html.Button(id="predict-btn", children="Predict", n_clicks=0),
            ),
            html.Br(),
            html.Br(),
            html.H3("Model features"),
            html.Div(
                id="info_card",
                children=[
                    html.B("Precision on test dataset"),
                    html.Hr(),
                    html.H5("16,07%", style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                    ]
            ),
            html.Br(),
            html.Div(
                id="nb_room_card",
                children=[
                    html.B("Recall on test dataset"),
                    html.Hr(),
                    html.H5("16,26%", style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                    ]
            ),
            html.Br(),
            html.Div(
                id="nb_ch_card",
                children=[
                    html.B("F1 Score on test dataset"),
                    html.Hr(),
                    html.H5("15.58%", style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                    ]
            ),
            html.Br(),
            html.Br(),
            html.H3("Class features"),
            html.Div(
                id="delay_card",
                children=[
                    html.B("Precision on this class"),
                    html.Hr(),
                    html.H5(id="feat_delay", style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                    ],
            ),
            html.Br(),
            html.Div(
                id="use_ch_card",
                children=[
                    html.B("Recall on this class"),
                    html.Hr(),
                    html.H5(id="feat_ch", style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                    ],
            ),
        ],
    )

# ====================================================
# App
# ====================================================

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("Logo_UPHF.png")),
                      html.Img(src=app.get_asset_url("Logo-INSA.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()],
        ),
        # Right column
        html.Div(
             id="right-column",
             className="eight columns",
             children=[
                html.Div(
                    id="box_card",
                    children=[
                        html.B("FaultStatus Result"),
                        html.Hr(),
                        html.H5(id="res_p", style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
                    ],
                ),
                html.Div(
                    id="hs_card",
                    children=[
                        html.B("Predictions"),
                        html.Hr(),
                        dcc.Graph(id="hist-chart"),
                    ],
                ),
                html.Div(
                    id="room_card",
                    children=[
                        html.B("Scatter plot"),
                        html.Hr(),
                        dcc.Graph(id="scatter-chart",
                                  figure = px.scatter_3d(df, x='Units', y='RPM', z='DataValue', color="FaultStatus")),
                    ],
                ),
                html.Div(
                    id="line_card",
                    children=[
                        html.B("Confusion matrix on test data"),
                        html.Hr(),
                        dcc.Graph(id="heat-chart",
                                  figure = px.imshow(confusion_mat,
                labels=dict(x="Predicted", y="Actual", color="Predictions"),
                x=['1. FOR INFORMATION', '2. NO INTERVENTION REQUIRED',
       '3. ATTENTIVE FOLLOW UP', '4. PROACTIVE INTERVENTION',
       '5. CORRECTIVE INTERVENTION'],
                y=['1. FOR INFORMATION', '2. NO INTERVENTION REQUIRED',
       '3. ATTENTIVE FOLLOW UP', '4. PROACTIVE INTERVENTION',
       '5. CORRECTIVE INTERVENTION']
               )),
                    ],
                ),
             ],
        ),
    ],
)

@app.callback([Output("res_p", "children"),
               Output("hist-chart", "figure"),
               Output("feat_ch", "children"),
               Output("feat_delay", "children")],
              [Input("predict-btn", "n_clicks")],
              [State("dtrue", "value"),
               State("datavalue", "value"),
               State("radio-select", "value"),
               State("eqclass-select", "value"),
               State("desarea-select", "value"),
               State("deseq-select", "value"),
               State("mptid-select", "value"),
               State("desmpt-select", "value"),
               State("desapset-select", "value"),
               State("desaparm-select", "value")]
              )
def update_hist_chart(n_clicks, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10):

    dict = {'EquipmentClassification' : v4,
            'Description_Area' : v5,
            'Description_Equipment' : v6,
            'MptID' : v7,
            'Description_MeasPt' : v8,
            'Description_APSet' : v9,
            'Description_AnalParm' : v10,
            'DataValue' : v2,
            'Units' : v3,
            'RPM' : v1,}

    cla = ['1. FOR INFORMATION', '2. NO INTERVENTION REQUIRED',
       '3. ATTENTIVE FOLLOW UP', '4. PROACTIVE INTERVENTION',
       '5. CORRECTIVE INTERVENTION']

    df_ = pd.DataFrame(dict, index=[0])

    proba = XGBC.predict_proba(df_)[0]

    # Define Figure
    data = []
    colors = ['lightslategray',] * 5
    colors[np.argmax(proba)] = 'crimson'
    # Create subplots: use 'domain' type for Pie subplot
    data.append(go.Bar(x=cla, y=proba, marker_color=colors))

    fig = go.Figure(
        {"data": data,})
    fig.update_layout(yaxis_title="Proba")

    #
    recall_res = np.round(recall_score(real, pred, average=None)[np.argmax(proba)], decimals=4)
    precision_res = np.round(precision_score(real, pred, average=None)[np.argmax(proba)], decimals=4)

    return cla[np.argmax(proba)], fig, str(recall_res*100)+'%', str(precision_res*100)+'%'

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)