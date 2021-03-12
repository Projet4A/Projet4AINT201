import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.title('P4A INSA Hauts-de-France')

@st.cache(suppress_st_warning=True)
def load_data(nrows):
    data = pd.read_csv(r'C:\Users\piotr\ClientA_Data_NC.csv',nrows=nrows)
    return data

data = load_data(27120)

equip=st.multiselect("Equipment",data.EquipmentClassification.unique())

if not equip:
    st.warning('Please input a equipment.')
    st.stop()

df = pd.crosstab(data.EquipmentClassification,data.FaultStatus).reset_index()
fig = px.bar(df[df.EquipmentClassification.isin(equip)], x=equip, y=data.FaultStatus.unique(), title="Wide-Form Input")

st.plotly_chart(fig)
