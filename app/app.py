import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from geopy.geocoders import ArcGIS
from datetime import date

st.set_page_config(page_title="House Price Prediction", layout="centered")
# Title
st.title("House Price Prediction App")
st.markdown("""Welcome to the **House Price Prediction App**!
            Enter property details in the sidebar to estimate the rent price """)
#st.write("Enter your Adresse to get the Latitude and longitude")


# get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'deployement')
 #load files using absolute paths
model = joblib.load(os.path.join(MODEL_DIR,'house_price_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR,'scaler.pkl'))
features_columns = joblib.load(os.path.join(MODEL_DIR,'features_columns.pkl'))


def clean_new_data(df):
    df[['yearConstructed','noParkSpaces','noRooms']]= df[['yearConstructed','noParkSpaces','noRooms']].astype(int)

    if (df['livingSpace'] <=0).sum() !=0:
        st.warning('Found and removed negative value in livingSpace Column\
                   for the analysis. you can proceed your own to fix error and \
                   reupload the file')
        df =df[df['livingSpace']>0]
    if (df['noRooms']< 0).sum() !=0:
        st.warning('Found negative or zero value must be \
                   at least 1. Is fill with 1 for better purpose of the analysis')
        df[df['noRooms'] <0 ] = 1
    
    if ('price_per_m2' not in df.columns) | ('age_of_house' not in df.columns):
        st.warning('Column not found. Also fill for the analysis')
        df['age_of_house'] = date.today().year- df['yearConstructed']
        df['price_per_m2'] = (df['baseRent']/ df['livingSpace']).round(4)
    
    if ('latitude' not in df.columns) | ('longitude' not in df.columns):
        st.warning("Missing columns")
        df['latitude'] = 50.80
        df['longitude'] = 10.30
    if ((~df['latitude'].between(-90,90).any()) | (~df['longitude'].between(-180,180).any())):
        st.error('check range of latitude and longitude latitude range [-90.0, 90.0], longitude range [-180.0 , 180.0]')
        st.stop()

    df = df[features_columns].copy()
            # scale numeric columns
    
    column_to_scale = ['yearConstructed', 'noRooms','livingSpace','latitude', 'longitude', 'price_per_m2','age_of_house'] 
    if df[column_to_scale].isnull().any().any():
        st.error("Your uploaded file contains missing or infinite values. Please clean it and upload again.")
        return

    df[column_to_scale] = scaler.transform(df[column_to_scale])
    return df




# Initialize the geolocator once
geolocator_arcgis = ArcGIS()

# --- Functions for clarity ---

def geocode_address(address):
    """Performs the geocoding and stores the result in session state."""
    try:
        # Perform the expensive operation
        location = geolocator_arcgis.geocode(address)
        if location:
            # Store the result in session state
            st.session_state['latitude'] = location.latitude
            st.session_state['longitude'] = location.longitude
            st.session_state['error'] = None # Clear any previous error
        else:
            st.session_state['error'] = "Address not found."
            st.session_state['latitude'] = None
            st.session_state['longitude'] = None

    except Exception as e:
        # Handle potential connection or API errors
        st.session_state['error'] = f"An error occurred: {e}"
        st.session_state['latitude'] = None
        st.session_state['longitude'] = None


# --- Streamlit UI ---

address = st.text_input(
    "Enter your Address to get the Latitude and Longitude:",
    "Walther-Nernst-Strasse 1, 38678 Clausthal-Zellerfeld, Germany",
    key="address_input" # Key allows us to easily get the value
)

# 2. Button to trigger the geocoding
# The callback function ensures geocoding only runs when the button is clicked
st.button(
    "Get Coordinates",
    on_click=geocode_address,
    args=(st.session_state.address_input,) # Pass the current input value to the function
)

# 3. Display the results from session state
st.write("---")
if 'error' in st.session_state and st.session_state.error:
    # Display error message if present
    st.error(st.session_state.error)

elif 'latitude' in st.session_state and st.session_state.latitude is not None:
    # Display results if available
    st.success("Coordinates Found!")
    st.write(f'**Latitude:** {st.session_state.latitude}')
    st.write(f'**Longitude:** {st.session_state.longitude}')
else:
    # Initial message or after an error reset
    st.info("Enter an address and click 'Get Coordinates' to start.")


st.sidebar.header("Enter Property Details")
yearConstructed = st.sidebar.number_input("Year Constructed",1900, 2025, 1990)
noRooms = st.sidebar.number_input("Number of Rooms", 1, 20, 3)
livingSpace = st.sidebar.number_input("Living Space (m^2)", 10.0, 500.0, 50.0)
noParkSpaces = st.sidebar.number_input("Parking Spaces", 0, 5, 1)
balcony = st.sidebar.checkbox("Balcony")
hasKitchen = st.sidebar.checkbox("Has Kitchen")
cellar = st.sidebar.checkbox("Cellar")
lift = st.sidebar.checkbox("Lift")
garden = st.sidebar.checkbox("Garden")
latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 51.8)
longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 10.3)
age_of_house = st.sidebar.number_input("Age of House", 0, 200, 30)
price_per_m2 = st.sidebar.number_input("Price per m^2 (€)", 5.0, 50.0, 20.0)

if st.sidebar.button("Predict Rent"):
    input_data = pd.DataFrame([{
       "yearConstructed":yearConstructed,
        "noRooms":noRooms,
        "livingSpace":livingSpace,
        "noParkSpaces":noParkSpaces,
        "balcony":balcony,
        "hasKitchen":hasKitchen,
        "cellar":cellar,
        "lift":lift,
        "garden":garden,
        "latitude":latitude,
        "longitude":longitude,
        "age_of_house":age_of_house,
        "price_per_m2":price_per_m2
    }])

    x = clean_new_data(input_data)
    prediction = model.predict(x)[0]

    #st.success(f"**Predict Rent:** € {prediction:,.2f}")
    st.metric("Predict Rent", "€{:,.2f}".format(prediction), border=True)
    st.table(input_data)
st.markdown(""" 
<style>
            .stApp{
            background-color: #f5f5f5;
            }
            h1 {
            color: #2E86C1;
            }
</style>
""", unsafe_allow_html=True)
files = st.file_uploader("Upload data", accept_multiple_files = False,type="csv")
if files is not None:

#for file in files:
    data = pd.read_csv(files)
    st.write(data)
    #st.download_button(label="Download CSV",data=data)
    x1 = clean_new_data(data)
    pred = model.predict(x1)[0]
    st.metric("Predict Rent", "€{:,.2f}".format(pred), border=True)
    #st.write(data)
 