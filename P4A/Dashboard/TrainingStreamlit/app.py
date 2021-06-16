import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from PIL import Image
import pathlib

# ====================================================
# Path
# ====================================================

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

st.title('P4A INSA Hauts-de-France')

@st.cache(suppress_st_warning=True)
def load_data(nrows):
    data = pd.read_csv(DATA_PATH.joinpath("ClientA_Data_NC.csv"),nrows=nrows)
    data.RPM.fillna(data.RPM.mean(),inplace=True)
    return data

data = load_data(27120)

st.write(data.head(5))

encodeur = LabelEncoder()
Y=encodeur.fit_transform(data.FaultStatus)
X=data.drop(['DateTime_x','Bearing_Location','Date','EquipmentID','SurveyDate','EquipmentName','AnalysisDate','FaultStatus','FaultName','Comment','Location','Analyst','AreaID','Route','EquipmentName'],axis=1)

size = st.slider('Size of the test set (%):',0.1,0.5,value=0.2,step=0.1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=size, random_state=42)

st.write('Shape of data_train:', X_train.shape)
st.write('Shape of data_test:', X_test.shape)

numerical_features = ['DataValue','RPM','Units']
categorical_features = ['EquipmentClassification','Description_Area','Description_Equipment','MptID','Description_MeasPt','Description_APSet','Description_AnalParm','AreaName']

numerical_pipeline = make_pipeline(StandardScaler())
categorical_pipeline = make_pipeline(OneHotEncoder(categories='auto',sparse=False,handle_unknown = 'ignore'))

transformer = make_column_transformer((numerical_pipeline, numerical_features),(categorical_pipeline,categorical_features))

def add_parameter():
    params = dict()
    n_estimators = st.sidebar.slider('N_estimators:', 1,100,value=50)
    params['n_estimators'] = n_estimators
    max_depth = st.sidebar.slider('Max_depth:', 2, 15, value=6)
    params['max_depth'] = max_depth
    learning_rate = st.sidebar.slider('Learning_rate:',0.01,0.5,value=0.20, step=0.01)
    params['learning_rate'] = learning_rate
    subsample = st.sidebar.slider('Subsample:',0.8,1.0,value=0.9,step=0.05)
    params['subsample'] = subsample
    colsample_bytree = st.sidebar.slider('Colsample_bytree:',0.1,1.0,value=1.0,step=0.1)
    params['colsample_bytree'] = colsample_bytree
    return params

params = add_parameter()

resp = st.radio('Train the model with this parameters ?',('Yes','No'),1)

if resp=='No':
    st.warning('Please select your parameters')
    st.stop()

@st.cache(suppress_st_warning=True,hash_funcs={xgb.XGBClassifier: id})
def train_classifier(params):
    clf=xgb.XGBClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],learning_rate=params['learning_rate'],subsample=params['subsample'],colsample_bytree = params['colsample_bytree'])
    model = make_pipeline(transformer,clf)
    model.fit(X_train, Y_train)
    return model

model = train_classifier(params)
st.write('Score of the model (train data):', model.score(X_train,Y_train))
st.write('Score of the model (test data):', model.score(X_test,Y_test))

save = st.radio('Save the model ?',('Yes','No'),1)
#
if save == 'No':
    st.stop()

name = st.text_input("Please give a file name (don't forget the extension .pkl)")

if not name:
    st.warning('Please give a file name if you want save the model')
    st.stop()
else:
    pickle.dump(model, open(name, "wb"))
    st.warning('The model is saved')

save_model(model)

#XBOOST Regressor Classifier (XGBC)
# name = st.sidebar.text_input('XGBOOST file name (add .pkl)')
# if not name:
#     st.warning('Please input a file name')
#     st.stop()

# file_name = "xgb_cla_fsall.pkl"

# load
# XGBC = pickle.load(open(name, "rb"))
#
# y_pred_xgbc = XGBC.predict(X_test)
# y_true_ = Y_test.ravel()
#
# recall_res_xgbc = recall_score(y_true_, y_pred_xgbc, average=None)
# precision_res_xgbc = precision_score(y_true_, y_pred_xgbc, average=None)
# f1score_res_xgbc = f1_score(y_true_, y_pred_xgbc, average=None)
# acc_res_xgbc = accuracy_score(y_true_, y_pred_xgbc)
#
# st.write('Classifier results')
# st.write(f'Accuracy =', acc_res_xgbc)
