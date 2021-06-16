import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.compose import make_column_transformer
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from xgboost import plot_importance
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_tree

# Load data
parse_dates = ['DateTime_x', 'Date', 'AnalysisDate','SurveyDate']

df = pd.read_csv(r'C:\Users\aurel\OneDrive\Documents\ENSIAME-INSA\Cours\S8\Projet4A\ClientA_Data_NC.csv', parse_dates=parse_dates)
df.RPM.fillna(df.RPM.mean(),inplace=True)

df.EquipmentClassification = df.EquipmentClassification.apply(lambda x: x.strip())

def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list

# Initialise the app
app = dash.Dash(__name__,title='P4AT INSA')

# Define the app
app.layout = html.Div(children=[

    html.Div(className='row', children=[
        html.Div(className='four columns div-user-controls',children = [

        html.H1('Choose your parameters'),
        html.H2('n_estimators'),
        html.Div([
            dcc.Slider(id='n_estimators', min=1, max=100, step=0.5,value=10,
                       marks={1: '1',100:'100'},),
            html.Div(id='n_estimators-output')
                ]),


        html.H2('Max_depth'),
        html.Div([
            dcc.Slider(id='Max_depth', min=2, max=15, step=1, value=1,
                       marks={2: '2',15:'15'},),
            html.Div(id='Max_depth-output')
                ]),


        html.H2('Learning_rate'),
        html.Div([
            dcc.Slider(id='Learning_Rate',min=0.01,max=0.5,step=0.01,value=0.20,
                        marks={0.01: '0.01',0.5:'0.5'},),
            html.Div(id='Learning_rate-output')
                ]),


        html.H2('Subsample'),
        html.Div([
            dcc.Slider(id='Subsample', min=0.8, max=1, step=0.01, value=0.8,
                    marks={0.8: '0.8',1:'1'},),
            html.Div(id='Subsample_rate-output')
                ]),


        html.H2('Colsample_bytree'),
        html.Div([
            dcc.Slider(id='Colsample_bytree',min=0.1,max=1,step=0.1,value=1,
                        marks={0.1: '0.1',1:'1'},),
            html.Div(id='Colsample_bytree_rate-output')
                ]),

]),


# Define the left element

    html.Div(className='eight columns div-for-charts bg-grey',children= [

        html.H1('DATAFRAME'),
        dash_table.DataTable( id='table',
                              columns=[{"name": i, "id": i} for i in df.columns],
                              data=df.to_dict('records'),
                              style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                              style_cell={'backgroundColor': 'rgb(50, 50, 50)','color': 'white'},
                              style_table={'height': '200px', 'overflowY': 'auto', 'overflowX': 'auto'},

                              ),

        html.H2('Size of the test set (%):'),
        html.Div([
            dcc.Slider(id='Test_size',min=0.1,max=0.5,step=0.1,value=0.2,
                        marks={0.1: '0.10',0.5:'0.5'},),
            html.Div(id='Test_size-output')
                ]),

        html.Div([
            html.Br(),
            html.Div(id='size-output'),
                ]),


        html.Div([
            html.H1('Train the model with the selected parameters?'),

            dcc.RadioItems(
                id='Train',
                options=[{'label': 'Yes', 'value': 'Y'},
                        {'label': 'No', 'value': 'N'},],
                        value='N'
                        )
            ]),


        html.H1('Score of the model'),
        html.Div([
            html.Br(),
            html.Div(id='model-output'),
                ]),

])
])
])

@app.callback(
    dash.dependencies.Output('size-output','children'),
    [dash.dependencies.Input('Test_size','value')])
def update_output_div(value):
    return '{}'.format(value)

@app.callback(
    dash.dependencies.Output('n_estimators-output', 'children'),
    [dash.dependencies.Input('n_estimators', 'value')])
def update_output(value):
    return '{}'.format(value)


@app.callback(
    dash.dependencies.Output('Max_depth-output', 'children'),
    [dash.dependencies.Input('Max_depth', 'value')])
def update_output(value):
    return '{}'.format(value)

@app.callback(
    dash.dependencies.Output('Learning_rate-output', 'children'),
    [dash.dependencies.Input('Learning_Rate', 'value')])
def update_output(value):
    return '{}'.format(value)

@app.callback(
    dash.dependencies.Output('Subsample_rate-output', 'children'),
    [dash.dependencies.Input('Subsample', 'value')])
def update_output(value):
    return '{}'.format(value)


@app.callback(
    dash.dependencies.Output('Colsample_bytree_rate-output', 'children'),
    [dash.dependencies.Input('Colsample_bytree', 'value')])
def update_output(value):
    return '{}'.format(value)

@app.callback(
    dash.dependencies.Output('model-output', 'children'),
    [dash.dependencies.Input('n_estimators', 'value'),
    dash.dependencies.Input('Max_depth', 'value'),
    dash.dependencies.Input('Learning_Rate', 'value'),
    dash.dependencies.Input('Colsample_bytree', 'value'),
    dash.dependencies.Input('Subsample', 'value'),
    dash.dependencies.Input('Test_size','value'),
    dash.dependencies.Input('Train','value')
    ])

def update_output(n_estimators,Max_depth,Learning_Rate,Colsample_bytree,Subsample,Test_size,Train):

    if Train=='Y':

        encodeur = LabelEncoder()
        Y=encodeur.fit_transform(df.FaultStatus)
        X=df.drop(['DateTime_x','Bearing_Location','Date','EquipmentID','SurveyDate','EquipmentName','AnalysisDate','FaultStatus','FaultName','Comment','Location','Analyst','AreaID','Route','EquipmentName'],axis=1)

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=Test_size, random_state=42)

        numerical_features = ['DataValue','RPM','Units']
        categorical_features =      ['EquipmentClassification','Description_Area','Description_Equipment','MptID','Description_MeasPt','Description_APSet','Description_AnalParm','AreaName']

        numerical_pipeline = make_pipeline(StandardScaler())
        categorical_pipeline = make_pipeline(OneHotEncoder(categories='auto',sparse=False,handle_unknown = 'ignore'))

        transformer = make_column_transformer((numerical_pipeline, numerical_features),(categorical_pipeline,categorical_features))

        clf=xgb.XGBClassifier(n_estimators=n_estimators,max_depth=Max_depth,learning_rate=Learning_Rate,subsample=Subsample,colsample_bytree =Colsample_bytree)
        model = make_pipeline(transformer,clf)
        model.fit(X_train, Y_train)

        return "Score of the model (train data):{}".format(model.score(X_train,Y_train)) + "\n" + "             Score of the model (test data):{}".format(model.score(X_test,Y_test))


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8000, host='127.0.0.1') #, use_reloader=False, dev_tools_hot_reload=False, port=8081)
