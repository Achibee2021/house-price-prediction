import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from geopy.geocoders import ArcGIS
from datetime import date
import shap
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

#----------1. CONFIGUTION & ASSETS  -----------

st.set_page_config(page_title="House Price Prediction", layout="wide")

# get the directory of the current file
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, '..', 'deployment')
    #load files using absolute paths
    model = joblib.load(os.path.join(MODEL_DIR,'house_price_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR,'scaler.pkl'))
    features_columns = joblib.load(os.path.join(MODEL_DIR,'features_columns.pkl'))

    explainer = shap.Explainer(model)

    return model, scaler, features_columns, explainer

model, scaler, features_columns, explainer = load_assets()

#------2. SESSION STATE INITIALIZATION-----
if "latitude" not in st.session_state:
    st.session_state.latitude = 51.8

if "longitude" not in st.session_state:
    st.session_state.longitude = 10.3

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "base_scenario" not in st.session_state:
    st.session_state.base_scenario = None


#-------- 3. HELPER FUNCTIONS ----------

def clean_new_data(df):
    """Process raw input to match model requirements."""

    df = df.copy()

    # Ensure types
    cols_to_fix = ['yearConstructed','noParkSpaces','noRooms']
    df[cols_to_fix]= df[cols_to_fix].fillna(0).astype(int)

    # Basic cleaning
    df =df[df['livingSpace'] > 0]
    df.loc[df['noRooms'] <=0, 'noRooms'] = 1
    
    if ('age_of_house' not in df.columns):
        df['age_of_house'] = date.today().year- df['yearConstructed']
    
    if ('latitude' not in df.columns) | ('longitude' not in df.columns):
        df.loc[:, 'latitude'] = 50.80
        df.loc[:, 'longitude'] = 10.30

    # Check bounds
    if not df['latitude'].between(-90, 90).all() or not df['longitude'].between(-180,180).all():
        st.error('Latitide/Longitide out of bounds!')
        st.stop()
    
    #Reorder columns to match trainer
    df = df[features_columns]

    # scale numeric columns
    column_to_scale = ['yearConstructed', 'noRooms','livingSpace','latitude', 'longitude','age_of_house'] 
    if df[column_to_scale].isnull().any().any():
        st.error("Your uploaded file contains missing or infinite values. Please clean it and upload again.")
        return

    df[column_to_scale] = scaler.transform(df[column_to_scale])
    return df

def geocode_address(address):
    geolocator_arcgis = ArcGIS()
    """Performs the geocoding and stores the result in session state."""
    try:
        # Perform the expensive operation
        location = geolocator_arcgis.geocode(address)
        if location:
            # Store the result in session state
            st.session_state.latitude= location.latitude
            st.session_state.longitude = location.longitude
            st.session_state.error = None # Clear any previous error
        else:
            st.session_state.error = "Address not found."

    except Exception as e:
        # Handle potential connection or API errors
        st.session_state.error = f"Geocoding error: {e}"

def simulate_price_trend(base_input, years_ahead=10):
    """Simunlate rent evolution as he house ages."""

    results = []
    current_year = date.today().year

    for i in range(-2, years_ahead + 1):
        year= current_year + i

        simulated = base_input.copy()
        simulated["age_of_house"] = base_input["age_of_house"] + i

        df = pd.DataFrame([simulated])
        df_sim = clean_new_data(df=df)

        if df_sim is not None:
            pred = model.predict(df_sim)[0]
            results.append({
                "year":year,
                "predicted_rent":pred
            })
    return pd.DataFrame(results).set_index("year")
        
# ------ 4. MAP CLICK LOGIC (Must be before Map Render) ---------

if "map_output" in st.session_state and st.session_state.map_output:
    last_click = st.session_state.map_output.get("last_clicked")
    if last_click:
        new_lat, new_lon = last_click["lat"], last_click["lng"]

        #Check if the click is actually NEW
        if new_lat != st.session_state.latitude or new_lon != st.session_state.longitude:
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon

            # ----- Get the address name from the click -----
            try:
                geolocator = ArcGIS()
                location = geolocator.reverse((new_lat, new_lon))
                if location:
                    st.session_state.address_input = location.address
            except:
                pass # Silently fail if reverse geocode fails

            st.rerun() # Force immediate update of the UI Marker


# ------------ 5. UI: LAYOUT --------
st.title("House Price Prediction App")
st.markdown("Estimate rent prices by entering details or clicking the map")
with st.expander("Search by Address"):
    address = st.text_input("Enter Adrdress:", "38678 Clausthal-Zellerfeld, Germany", key="address_input")
    if st.button("Find on Map"):
        geocode_address(address)
        
col_main, col_side = st.columns(2)

with col_main:
    st.subheader("Select Location")
    if "address_input" in st.session_state:
        st.info(f"Selected: {st.session_state.address_input}")

    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=12)

    folium.Marker(
        location=[st.session_state.latitude, st.session_state.longitude],
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    # Store Output in Session state to catch on next rerun
    st.session_state.map_output = st_folium(
        m, 
        width="100%", 
        height=400, 
        key="selector_map")

with st.sidebar:
    with st.form("input_form"):
        st.header("Enter Property Details")
        yearConstructed = st.number_input("Year Constructed",1900, 2025, 1990)
        noRooms = st.number_input("Number of Rooms", 1, 20, 3)
        livingSpace = st.number_input("Living Space (m^2)", 10.0, 500.0, 50.0)
        noParkSpaces = st.number_input("Parking Spaces", 0, 5, 1)

        col1, col2 = st.columns(2)
        with col1:
            balcony = st.checkbox("Balcony")
            hasKitchen = st.checkbox("Has Kitchen")
        with col2:
            cellar = st.checkbox("Cellar")
            lift = st.checkbox("Lift")
        garden = st.checkbox("Garden")

        # Pulling coordinates from session state (MAP)

        lat_input = st.number_input("Latitude", value=st.session_state.latitude, format="%.4f")
        lon_input = st.number_input("Longitude", value=st.session_state.longitude, format="%.4f")

        submit_button = st.form_submit_button("Predict Rent")


# ------ 6. PREDICTION ------------

if submit_button:
    st.session_state.base_scenario ={
        "yearConstructed":yearConstructed,
        "noRooms":noRooms,
        "livingSpace":livingSpace,
        "noParkSpaces":noParkSpaces,
        "balcony":balcony,
        "hasKitchen":hasKitchen,
        "cellar":cellar,
        "lift":lift,
        "garden":garden,
        "latitude":lat_input,
        "longitude":lon_input,
        "age_of_house":(date.today().year - yearConstructed)
    }

    st.session_state.show_results = True

# ------- 7. PERSISTENT RESULTS DISPLAY -----------

if st.session_state.show_results:
        st.divider()
        current_data = st.session_state.base_scenario.copy()

        # SCENARION SIMULATOR
        st.subheader("Live Scenario Simulator")
        st.write("Toogle features below to see how they impact therent and trend.")

        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            sim_balcony = st.checkbox("Add Balcony", value=current_data["balcony"])
        with sim_col2:
            sim_kitchen = st.checkbox("Modern Kitchen", value=current_data["hasKitchen"])
        with sim_col3:
            sim_garden = st.checkbox("Private Garden", value=current_data["garden"])

        #update data based on simulation
        current_data.update({"balcony": sim_balcony, "hasKitchen":sim_kitchen, "garden":sim_garden})

        #final Prediction for Metrcis
        x_final = clean_new_data(pd.DataFrame([current_data]))
        final_pred = model.predict(x_final)[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Total Rent", f"€{final_pred:,.2f}")
        m2.metric("Price per m2", f"€{(final_pred/current_data['livingSpace']):,.2f}")
        m3.metric("Property Age", f"{current_data['age_of_house']} Years")

        # VISUALS
        tab1, tab2 = st.tabs(["Price Trend", "Feature Importance (SHAP)"])

        with tab1:
            st.write("### Simulated Rent Rvolution")
            trend_data = simulate_price_trend(current_data)
            st.line_chart(trend_data)

        with tab2:
            st.write("#### What drives this price?")
            shap_values = explainer(x_final)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.clf()

# ---- 8. BATCH UPLOAD ----------
st.divider()
with st.expander("Batch Processing (CSV)"):
        uploader = st.file_uploader("Upload property list", type="csv")
        if uploader:
            df_uploaded = pd.read_csv(uploader)
            st.success("File uploaded! Proceed to prediction logic.")
# Custom
st.markdown(
    "<style>h1 {color: #2E86C1;} .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ddd;}</style>", unsafe_allow_html=True)