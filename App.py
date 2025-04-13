import fastf1
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# ========== CACHE SETUP ==========
try:
    # Try to create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    fastf1.Cache.enable_cache('cache')
except Exception as e:
    st.warning(f"Cache setup failed: {e}. Continuing without cache...")

# Configure Streamlit
st.set_page_config(layout="wide")
st.title("F1 2025 Ultimate Race Analyzer")

# ========== SESSION SELECTION ==========
year = 2025
event = st.selectbox("Event", ["Australia", "Japan", "Monza"])
session_type = st.selectbox("Session", ["Race", "Qualifying"])

# ========== DATA LOADING ==========
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_session_data(year, event, session_type):
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        return session
    except Exception as e:
        st.error(f"Failed to load session data: {e}")
        return None

session = load_session_data(year, event, session_type)
if session is None:
    st.stop()

# Get all drivers
drivers = session.drivers
driver1 = st.selectbox("Driver 1", drivers)
driver2 = st.selectbox("Driver 2", [d for d in drivers if d != driver1])

# ========== GHOST RACING SIMULATOR ==========
def ghost_simulator(session, driver1, driver2):
    try:
        # Get laps with tire data
        lap1 = session.laps.pick_driver(driver1).pick_fastest()
        lap2 = session.laps.pick_driver(driver2).pick_fastest()
        
        # Get telemetry
        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()
        
        # Create figure
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=(
                               f"Speed Comparison | {driver1} ({lap1['Compound']}) vs {driver2} ({lap2['Compound']})",
                               "Throttle/Brake",
                               "Delta Time", 
                               "Tire Age Comparison"))
        
        # Speed trace
        fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Speed'], 
                       name=f"{driver1} Speed", line=dict(color='red')), 1, 1)
        fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Speed'], 
                       name=f"{driver2} Speed", line=dict(color='blue')), 1, 1)
        
        # Throttle/Brake
        fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Throttle'], 
                       name=f"{driver1} Throttle", line=dict(color='orange')), 2, 1)
        fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Brake'], 
                       name=f"{driver1} Brake", line=dict(color='black')), 2, 1)
        
        # Delta time
        delta = tel1['Time'] - tel2['Time']
        fig.add_trace(go.Scatter(x=tel1['Distance'], y=delta, 
                       name="Delta Time", line=dict(color='green')), 3, 1)
        
        # Tire age comparison
        if session_type == "Race":
            fig.add_trace(go.Scatter(x=tel1['Distance'], y=[lap1['TyreLife']]*len(tel1),
                                   name=f"{driver1} Tire Age", line=dict(color='darkred')), 4, 1)
            fig.add_trace(go.Scatter(x=tel2['Distance'], y=[lap2['TyreLife']]*len(tel2),
                                   name=f"{driver2} Tire Age", line=dict(color='darkblue')), 4, 1)
        
        # Overtaking spots
        crossings = np.where(np.diff(np.sign(delta)))[0]
        for x in crossings:
            fig.add_vline(x=tel1['Distance'].iloc[x], line_dash="dot", 
                          annotation_text="OVT Zone", row=1, col=1)
        
        fig.update_layout(height=1000, showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Ghost simulator error: {e}")
        return go.Figure()

# ========== MAIN DASHBOARD ==========
st.header("👻 Ghost Race Simulator")
ghost_fig = ghost_simulator(session, driver1, driver2)
st.plotly_chart(ghost_fig, use_container_width=True)

# ========== TIRE ANALYSIS ==========
st.header("🔄 Tire Strategy Analysis")
try:
    # Get all laps with tire data
    laps = session.laps
    tire_data = laps[['Driver', 'LapNumber', 'Compound', 'TyreLife', 'LapTime']].dropna()
    
    # Plot tire performance degradation
    fig = go.Figure()
    for driver in tire_data['Driver'].unique():
        driver_data = tire_data[tire_data['Driver'] == driver]
        fig.add_trace(go.Scatter(x=driver_data['TyreLife'], y=driver_data['LapTime'],
                                mode='markers+lines', name=driver,
                                marker=dict(symbol=driver_data['Compound'].map(
                                    {'SOFT':'circle', 'MEDIUM':'square', 'HARD':'diamond'}))))
    
    fig.update_layout(title="Lap Time vs Tire Age",
                     xaxis_title="Tire Age (laps)",
                     yaxis_title="Lap Time (s)")
    st.plotly_chart(fig)
    
    # Show tire usage table
    st.subheader("Tire Usage Summary")
    st.dataframe(tire_data.groupby(['Driver', 'Compound']).agg(
        {'LapNumber':'count', 'LapTime':'mean'}).rename(
        columns={'LapNumber':'Laps', 'LapTime':'Avg Lap Time'}))
except Exception as e:
    st.error(f"Tire analysis error: {e}")

# ========== WEATHER DATA ==========
st.header("🌦️ Session Weather")
try:
    st.write(session.weather_data)
except Exception as e:
    st.error(f"Weather data error: {e}")
