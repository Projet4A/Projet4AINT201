###Imports

import numpy as np
import pandas as pd
import datetime

df = pd.read_csv(r'C:\Users\piotr\ClientA_Reports.csv',converters={'SurveyDate':pd.to_datetime,'AnalysisDate':pd.to_datetime}) #Lecture du fichier et convertion de 'SurveyDate' et 'AnalysisDate' en datetime.

df['Date'] = df.AnalysisDate.apply(lambda x: x.date()) #Nouveau nom pour les donnees 'AnalysisDate' -> 'Date.
df['Description_Equipment'] = df.EquipmentName.apply(lambda x: "".join(x.strip().split()[1:]))
df.drop('Unnamed: 0',axis=1, inplace=True)
df.to_csv('ClientA_Reports_.csv',index=False)

data = pd.read_csv(r'C:\Users\piotr\ClientA_Vib_Data.csv', converters={'DateTime_x':pd.to_datetime}) #Lecture du fichier et convertion de 'DateTime' en datetime.

data['Description_Equipment'] = data.Description_Equipment.apply(lambda x: "".join(x.strip().split()))
data.drop('Unnamed: 0',axis=1, inplace=True)

def between(date, equipement):
    d=df[df.Description_Equipment==equipement]
    for i in range(len(d)):
        if ((date>d.SurveyDate.iloc[i])and(date<d.AnalysisDate.iloc[i])):
            return d.AnalysisDate.iloc[i].date()
            break

data['Date'] = data.apply(lambda x: between(x.DateTime_x,x.Description_Equipment),axis=1)
data.to_csv('ClientA_Vib_Data_.csv',index=False)

data.merge(df, how='inner',on=['Date','Description_Equipment'])

