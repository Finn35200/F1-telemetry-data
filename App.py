import fastf1
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ========== CACHE SETUP ==========
try:
    os.makedirs('cache', exist_ok=True)
    fastf1.Cache.enable_cache('cache')
except Exception as e:
    st.warning(f"Cache setup failed: {e}. Continuing without cache...")

# Configure Streamlit
st.set_page_config(layout="wide")
st.title("F1 Ultimate Race Analyzer")

# ========== SESSION SELECTION ==========
current_year = datetime.now().year
year = st.number_input("Year", min_value=2018, max_value=current_year, value=current_year)

# Get all events for selected year
try:
    schedule = fastf1.get_event_schedule(year)
    events = schedule.EventName.tolist()
    event = st.selectbox("Event", events)
except:
    st.error("Could not load events for selected year")
    st.stop()

session_type = st.selectbox("Session", ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race", "Sprint", "Sprint Qualifying"])

# ========== DATA LOADING ==========
@st.cache_data(ttl=3600)
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

# Get all drivers with full names
drivers = pd.unique(session.laps[['DriverNumber', 'Driver']].dropna().apply(
    lambda x: f"{x['DriverNumber']} - {x['Driver']}", axis=1))
driver1 = st.selectbox("Driver 1", drivers)
driver2 = st.selectbox("Driver 2", [d for d in drivers if d != driver1])

# Extract driver numbers for processing
driver1_num = driver1.split(" - ")[0]
driver2_num = driver2.split(" - ")[0]

# ========== GHOST RACING SIMULATOR ==========
def ghost_simulator(session, driver1_num, driver2_num):
    try:
        # Get laps with tire data
        lap1 = session.laps.pick_driver(driver1_num).pick_fastest()
        lap2 = session.laps.pick_driver(driver2_num).pick_fastest()
        
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
        if 'TyreLife' in lap1 and 'TyreLife' in lap2:
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
ghost_fig = ghost_simulator(session, driver1_num, driver2_num)
st.plotly_chart(ghost_fig, use_container_width=True)

# ========== TIRE ANALYSIS ==========
st.header("🔄 Tire Strategy Analysis")
try:
    # Get all laps with tire data
    laps = session.laps
    tire_data = laps[['DriverNumber', 'Driver', 'LapNumber', 'Compound', 'TyreLife', 'LapTime']].dropna()
    
    # Clean compound data
    tire_data['Compound'] = tire_data['Compound'].str.upper().str.strip()
    valid_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    tire_data = tire_data[tire_data['Compound'].isin(valid_compounds)]
    
    # Map compounds to symbols with fallback
    compound_symbols = {
        'SOFT': 'circle',
        'MEDIUM': 'square',
        'HARD': 'diamond',
        'INTERMEDIATE': 'triangle-up',
        'WET': 'x'
    }
    tire_data['Symbol'] = tire_data['Compound'].map(compound_symbols).fillna('circle')
    
    # Plot tire performance degradation
    fig = go.Figure()
    for driver in tire_data['Driver'].unique():
        driver_data = tire_data[tire_data['Driver'] == driver]
        fig.add_trace(go.Scatter)
            x=driver_data['TyreLife'],
            y=driver_data['LapTime'],
            mode='markers+lines',
            name=driver,
            marker=dict(
                symbol=driver_data['Symbol'],
                size=8,
                line=dict(width=1, color='DarkSlateGrey')
            )
        ))
    
    fig.update_layout(
        title="Lap Time vs Tire Age",
        xaxis_title="Tire Age (laps)",
        yaxis_title="Lap Time (s)",
        hovermode='closest'
    )
    st.plotly_chart(fig)
    
    # Show tire usage table
    st.subheader("Tire Usage Summary")
    summary = tire_data.groupby(['Driver', 'Compound']).agg(
        Laps=('LapNumber', 'count'),
        AvgLapTime=('LapTime', 'mean'),
        BestLapTime=('LapTime', 'min')
    ).reset_index()
    st.dataframe(summary.sort_values(['Driver', 'Laps'], ascending=[True, False]))

except Exception as e:
    st.error(f"Tire analysis error: {e}")

# ========== WEATHER DATA ==========
st.header("🌦️ Session Weather")
try:
    if hasattr(session, 'weather_data'):
        weather = session.weather_data
        st.write(f"Air Temp: {weather['AirTemp'].mean():.1f}°C | Track Temp: {weather['TrackTemp'].mean():.1f}°C")
        st.write(f"Humidity: {weather['Humidity'].mean():.1f}% | Rainfall: {'Yes' if weather['Rainfall'].any() else 'No'}")
        
        fig_weather = go.Figure()
        fig_weather.add_trace(go.Scatter(
            x=weather['Time'],
            y=weather['AirTemp'],
            name='Air Temp',
            line=dict(color='red')
        ))
        fig_weather.add_trace(go.Scatter(
            x=weather['Time'],
            y=weather['TrackTemp'],
            name='Track Temp',
            line=dict(color='orange')
        ))
        fig_weather.update_layout(
            title="Temperature During Session",
            xaxis_title="Session Time",
            yaxis_title="Temperature (°C)"
        )
        st.plotly_chart(fig_weather)
    else:
        st.warning("No weather data available for this session")
except Exception as e:
    st.error(f"Weather data error: {e}")

# ========== LAP TIME SUMMARY ==========
st.header("⏱️ Lap Time Analysis")
try:
    laps = session.laps
    valid_laps = laps[laps['LapTime'].notna()]
    
    if len(valid_laps) > 0:
        fig_laps = go.Figure()
        for driver in valid_laps['Driver'].unique():
            driver_laps = valid_laps[valid_laps['Driver'] == driver]
            fig_laps.add_trace(go.Scatter(
                x=driver_laps['LapNumber'],
                y=driver_laps['LapTime'],
                mode='markers+lines',
                name=driver
            ))
        
        fig_laps.update_layout(
            title="Lap Times During Session",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (s)",
            hovermode='closest'
        )
        st.plotly_chart(fig_laps)
    else:
        st.warning("No lap time data available")
except Exception as e:
    st.error(f"Lap time analysis error: {e}")
