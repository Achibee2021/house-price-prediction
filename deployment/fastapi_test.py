from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List
import os

# load all varaible to use 
# get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load files using absolute paths
model = joblib.load(os.path.join(BASE_DIR, 'house_price_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
features_columns = joblib.load(os.path.join(BASE_DIR, 'features_columns.pkl'))
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


app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "API running"}

def clean_new_data(df):

    #extract just the column to work with
    df = df[features_columns].copy()

    # scale numeric columns
    column_to_scale = ['yearConstructed', 'noRooms','livingSpace','latitude', 'longitude','age_of_house'] 
    df[column_to_scale] = scaler.transform(df[column_to_scale])
    return df


# Post for single prediction
@app.post("/predict")
def predict(features:HouseFeatures):
    

    df = pd.DataFrame([features.dict()])

    x = clean_new_data(df)

    y_pred = model.predict(x)[0]

    return {'predict_rent': float(y_pred)}

# Post for batch Predictions
@app.post("/predict_batch")
def prediction_batch(features:List[HouseFeatures]):
    data = [feature.dic() for feature in features]

    df = pd.DataFrame(data)

    x = clean_new_data(df)

    predictions = model.predict(x)

    return {"predicted_rent": predictions.tolist()}

