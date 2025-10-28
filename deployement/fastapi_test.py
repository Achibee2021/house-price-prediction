from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List
# load all varaible to use 

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
features_columns = joblib.load('features_columns.pkl')

# Define Base Model
class HouseFeatures(BaseModel):
    yearConstructed:float
    noRooms:int
    livingSpace:float
    noParkSpaces:int
    balcony:bool
    hasKitchen:bool
    cellar:bool	
    lift:bool	
    garden:bool
    latitude:float	
    longitude:float
    age_of_house:int
    price_per_m2:float


app = FastAPI()

def clean_new_data(df):

    #extract just the column to work with
    df = df[features_columns].copy()

    # scale numeric columns
    column_to_scale = ['yearConstructed', 'noRooms','livingSpace','latitude', 'longitude', 'price_per_m2','age_of_house'] 
    df[column_to_scale] = scaler.transform(df[column_to_scale])
    return df


# Post for single prediction
@app.post("/predict")
def prediction(features:HouseFeatures):
    

    df = pd.DataFrame([features.dict()])

    x = clean_new_data(df)
    y_pred = model.predict(x)
    return ({'predict_rent': float(y_pred[0])})

# Post for batch Predictions
@app.post("/predict_batch")
def prediction(features:List[HouseFeatures]):
    dic = {}
    for feature in features:

        df = pd.DataFrame([feature.dict()])

        x = clean_new_data(df)
        y_pred = model.predict(x)
        res = float(y_pred[0])
        dic.setdefault('predicted_rent', []).append(res)
    return dic