import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc

# app = dash.Dash()
# colors = {
#     'background': '#111111',
#     'text': '#7FDBFF'
# }
# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='Hello Dash',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),
#     html.Div(children='Dash: A web application framework for Python.', style={
#         'textAlign': 'center',
#         'color': colors['text']
#     }),
#     dcc.Graph(
#         id='Graph1',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'plot_bgcolor': colors['background'],
#                 'paper_bgcolor': colors['background'],
#                 'font': {
#                     'color': colors['text']
#                 }
#             }
#         }
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)

# df = pd.read_csv(
#     'https://gist.githubusercontent.com/chriddyp/' +
#     '5d1ea79569ed194d432e56108a04d188/raw/' +
#     'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
#     'gdp-life-exp-2007.csv')
#
#
# app.layout = html.Div([
#     dcc.Graph(
#         id='life-exp-vs-gdp',
#         figure={
#             'data': [
#                 go.Scatter(
#                     x=df[df['continent'] == i]['gdp per capita'],
#                     y=df[df['continent'] == i]['life expectancy'],
#                     text=df[df['continent'] == i]['country'],
#                     mode='markers',
#                     opacity=0.8,
#                     marker={
#                         'size': 15,
#                         'line': {'width': 0.5, 'color': 'white'}
#                     },
#                     name=i
#                 ) for i in df.continent.unique()
#             ],
#             'layout': go.Layout(
#                 xaxis={'type': 'log', 'title': 'GDP Per Capita'},
#                 yaxis={'title': 'Life Expectancy'},
#                 margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#                 legend={'x': 0, 'y': 1},
#                 hovermode='closest'
#             )
#         }
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server()

# app.layout = html.Div([
#     dcc.Input(id='my-id', value='Dash App', type='text'),
#     html.Div(id='my-div'),
#     dcc.Dropdown(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value='MTL'
#     ),
#     dcc.Dropdown(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value=['MTL', 'SF'],
#         multi=True
#     ),
#     html.Label('Radio Items'),
#     dcc.RadioItems(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value='MTL'
#     ),
#     html.Label('Checkboxes'),
#     dcc.Checklist(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ]
#     ),
#     html.Label('Text Box'),
#     dcc.Input(value='MTL', type='text')
# ])
#
# @app.callback(
#     Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')]
# )
# def update_output_div(input_value):
#     return 'You\'ve entered "{}"'.format(input_value)
#
# if __name__ == '__main__':
#     app.run_server(debug=True)

# Load data
parse_dates = ['DateTime_x', 'Date', 'AnalysisDate','SurveyDate']

df = pd.read_csv(r'C:\Users\piotr\ClientA_Data_NC.csv', parse_dates=parse_dates)
df.RPM.fillna(df.RPM.mean(),inplace=True)

df.EquipmentClassification = df.EquipmentClassification.apply(lambda x: x.strip())

def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list

# Initialise the app
app = dash.Dash(__name__,title='P4A INSA')

# Define the app
app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls',children = [
    html.H1('Client A - EQUIPMENT'),
    html.H2('TIME SERIES'),
    html.P('''Visualising time series'''),
    html.P('''Pick one or more equipments from the dropdown below.'''),
    html.Div(className='div-for-dropdown',
          children=[
              dcc.Dropdown(id='stockselector',
                           options=get_options(df.EquipmentClassification.unique()),
                           multi=True,
                           value=df.EquipmentClassification.sort_values().unique()[0:2],
                           style={'backgroundColor': '#1E1E1E'},
                           className='stockselector')
                    ],
          style={'color': '#1E1E1E'}),
    html.H2('PIE CHART'),
    html.P('''Pick one equipment from the dropdown below.'''),
    html.Div(className='div-for-dropdown',
          children=[
        dcc.Dropdown(id='equipselector',
            options = get_options(df.EquipmentClassification.unique()),
            multi=False,
            value='MOTEUR',
            style={'backgroundColor': '#1E1E1E'},
            className='stockselector')],
        style={'color': '#1E1E1E'}),
    html.H2('HISTOGRAM'),
    html.P('''Pick one or more faultstatus from the dropdown below.'''),
    html.Div(className='div-for-dropdown',
          children=[
              dcc.Dropdown(id='fsselector',
                           options=get_options(df.FaultStatus.unique()),
                           multi=True,
                           value=df.FaultStatus.sort_values().unique()[0:2],
                           style={'backgroundColor': '#1E1E1E'},
                           className='stockselector')
                    ],
          style={'color': '#1E1E1E'}),
    html.Div(
    dcc.Graph(
        id='',
        figure={
            'data': [go.Bar(x=df.FaultStatus.value_counts().index, y=df.FaultStatus.value_counts().values, name='FaultStatus',orientation="v",marker_color=px.colors.qualitative.Plotly[:len(df.FaultStatus.value_counts())])
            ],
            'layout': go.Layout(
                  colorway=px.colors.qualitative.Plotly,
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'FaultStatus'})
        }
    ))
]),  # Define the left element
    html.Div(className='eight columns div-for-charts bg-grey',children=
    [dcc.Graph(id='timeseries',
          config={'displayModeBar': False}
                                    ),
    dcc.Graph(id="pie-chart"),
    dcc.Graph(id="hist-marg")]
        )])])

@app.callback(Output('timeseries', 'figure'),
              [Input('stockselector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    trace = []
    df_sub = pd.crosstab(df['DateTime_x'],df.EquipmentClassification).resample('M').sum()
    # STEP 2
    # Draw and append traces for each stock
    for stock in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub.index,
                                 y=df_sub[stock].values,
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center'))
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    fig = {'data': data,
              'layout': go.Layout(
                  colorway=px.colors.qualitative.Plotly,
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Occurrence of registered equipment-related incidents', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]})
              }

    return fig

@app.callback(Output("pie-chart", "figure"),
              [Input("equipselector", "value")])
def update_pie_chart(selected_dropdown_value):

    df_sub = df[df.EquipmentClassification==selected_dropdown_value].FaultStatus.value_counts().reset_index()

    df_sub_ = df[df.EquipmentClassification==selected_dropdown_value].FaultName.value_counts().reset_index()

    # Define Figure

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=df_sub['index'].values, values=df_sub.FaultStatus.values, name="FaultStatus"), 1, 1)
    fig.add_trace(go.Pie(labels=df_sub_['index'].values, values=df_sub_.FaultName.values, name="FaultName"),1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(colorway=px.colors.qualitative.Plotly,
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  autosize=True,
                  title={'text': 'FaultStatus and FaultName for {}'.format(selected_dropdown_value), 'font': {'color': 'white'}, 'x': 0.5})

    # fig = go.Figure(data=[go.Pie(labels=df_sub['index'].values, values=df_sub.FaultStatus.values)],layout=go.Layout(
    #               colorway=px.colors.qualitative.Plotly,
    #               template='plotly_dark',
    #               paper_bgcolor='rgba(0, 0, 0, 0)',
    #               plot_bgcolor='rgba(0, 0, 0, 0)',
    #               autosize=True,
    #               title={'text': 'FaultStatus for {}'.format(selected_dropdown_value), 'font': {'color': 'white'}, 'x': 0.5}))
    return fig

@app.callback(Output("hist-marg", "figure"),
              [Input("fsselector", "value")])
def update_hist_chart(selected_dropdown_value):

    # STEP 1
    trace = []
    # STEP 2
    # Draw and append traces for each stock
    for stock in selected_dropdown_value:
        trace.append(go.Histogram(x=df[df.FaultStatus==stock].RPM.values,xbins=dict( # bins used for histogram
        start=0.0,
        end=6000.0,
        size=1000
    ),name=stock))
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    fig = {'data': data,
              'layout': go.Layout(
                  colorway=px.colors.qualitative.Plotly,
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  xaxis_title="RPM",
                  title={'text': 'Occurrence of RPM values', 'font': {'color': 'white'}, 'x': 0.5})
              }
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)