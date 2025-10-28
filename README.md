# House Price Prediction

This Project predicts house rental prices in Germany using three major models **RandomForest**, **Catboost**, and **XGBoost**- regressor.
It includes the **FastAPI Application** that allows both single and batch predictions based on house features.

## Folder Structure

HousePricePrediction/
data/
immo_data.csv original data
immo_data_with_lat_lon.csv Datasets with latitude and longitude/
deployement/
fastapi_test.py main datei
features_columns.pkl
house_price_model.pkl
scaler.pkl
notebooks/
housepricepredictiveCleaning.ipynb
housepricepredictionanalyse.ipynb
requirements.txt python dependencies
README:md projet description

## Installation

1. clone the repository

git clone https://github.com/Achibee2021/HousePricePrediction.git
cd HousPricePrediction

2. create and Activate a virtual environment (optional but recommended)

# Windows

python -m venv venv
venv\Scripts\activate

# Linux

python -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Running the API

uvicorn deployement.fastapi_test:app --reload

this runs by default at:
http://127.0.0.1:8000

# Example Request

1. Single Prediction
   POST json to /predict
   {
   "yearConstructed": 1968,
   "noRooms": 3,
   "livingSpace": 40.2,
   "noParkSpaces": 2,
   "balcony": true,
   "hasKitchen": true,
   "cellar": true,
   "lift": true,
   "garden": true,
   "latitude": 51.8077,
   "longitude": 10.3384,
   "age_of_house": 57,
   "price_per_m2": 20.8
   }
2. Batch Prediction
   POST to /predict_batch
   [
   {
   "yearConstructed": 1968,
   "noRooms": 3,
   "livingSpace": 40.2,
   "noParkSpaces": 2,
   "balcony": true,
   "hasKitchen": true,
   "cellar": true,
   "lift": true,
   "garden": true,
   "latitude": 51.8077,
   "longitude": 10.3384,
   "age_of_house": 57,
   "price_per_m2": 20.8
   },
   {
   "yearConstructed": 2005,
   "noRooms": 4,
   "livingSpace": 80.5,
   "noParkSpaces": 1,
   "balcony": false,
   "hasKitchen": true,
   "cellar": false,
   "lift": false,
   "garden": true,
   "latitude": 51.8085,
   "longitude": 10.3390,
   "age_of_house": 20,
   "price_per_m2": 18.5
   }
   ]
