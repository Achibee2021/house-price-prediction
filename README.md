# House Price Prediction & Investment Simulator

An Interactive Real Estate Analytics tool built with **streamlit** and **Machine Learning**. This app predicts montly rental prices n Germany and allows users to simulate how property improvements (like adding a balcony or gatden ) affect market value over time.
It includes the **FastAPI Application** that allows both single and batch predictions based on house features.

## Key Features

**Map-Based Selection:** Integrated `folium` map with reverse-geocoding. Click anywhere to instantly fect coordinates and address details.
**Scenario Simulato:** A "What-If" engine that allows you to toggle property fetures (Lift, Garden, Kitchen) to see live updates on price and future trends.
**Explainable AI (SHAP):** Transparency is key. We SHAP waterfall plots to show exactly which features (Location, Age, Space) are driving the specific price for each property.
**Time-Series Trend Analytics:** Visualizes how the rent of the selected property is projected to evolve as the building ages.
**Persistent State Managment:** Optimized with Streanlit Session State to ensure a smooth, lag-free user experience during complex simulations.

## Technology Stack

**Frontend:** Streamlit, Folium
**Data Science:** Pandas, Numpy, Scikit-Learn
**Explainabiliy:** SHAP (Shapley Additive exPlanations)
**Geospatial:** Goepy (ArcGIS API)
**Deployment:** Docker, Render

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

# Run the App

streamlit run app/app.py
