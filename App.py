import fastf1
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure
fastf1.Cache.enable_cache('cache')
st.set_page_config(layout="wide")

# Title
st.title("F1 2025 Ultimate Race Analyzer")

# Session selection
year = 2025
event = st.selectbox("Event", ["Australia", "Japan", "Monza"])
session_type = st.selectbox("Session", ["Race", "Qualifying"])

# Load session
session = fastf1.get_session(year, event, session_type)
session.load()

# Get all drivers
drivers = session.drivers
driver1 = st.selectbox("Driver 1", drivers)
driver2 = st.selectbox("Driver 2", [d for d in drivers if d != driver1])

# ========== GHOST RACING SIMULATOR ==========
def ghost_simulator(session, driver1, driver2):
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

# ========== OVERTAKING PREDICTION ==========
def predict_overtaking_spots(session, driver1, driver2):
    # Prepare data with tire info
    laps = session.laps
    tel1 = laps.pick_driver(driver1).pick_fastest().get_telemetry().add_distance()
    tel2 = laps.pick_driver(driver2).pick_fastest().get_telemetry().add_distance()
    
    # Align telemetry
    merged = pd.merge_asof(tel1, tel2, on='Distance', 
                          suffixes=('_1', '_2'))
    
    # Features including tire difference
    merged['TireDelta'] = 0
    if 'Compound_1' in merged.columns and 'Compound_2' in merged.columns:
        # Simple tire advantage model (Soft > Medium > Hard)
        tire_rank = {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1}
        merged['TireDelta'] = merged['Compound_1'].map(tire_rank) - merged['Compound_2'].map(tire_rank)
    
    X = merged[['Speed_1', 'Speed_2', 'Throttle_1', 'Brake_1', 'TireDelta']]
    y = (merged['Speed_1'] > merged['Speed_2']).astype(int)
    
    # Train model
    model = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    
    # Predict
    merged['Overtaking_Prob'] = model.predict(X)
    hotspots = merged[merged['Overtaking_Prob'] > 0.7]['Distance'].unique()
    
    return hotspots

# ========== TIRE STRATEGY ANALYSIS ==========
def tire_analysis(session):
    st.header("🔄 Tire Strategy Analysis")
    
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

# ========== MAIN DASHBOARD ==========
st.header("👻 Ghost Race Simulator")
ghost_fig = ghost_simulator(session, driver1, driver2)
st.plotly_chart(ghost_fig, use_container_width=True)

st.header("🔥 Predicted Overtaking Spots")
hotspots = predict_overtaking_spots(session, driver1, driver2)
st.write(f"**{driver1}** can overtake **{driver2}** at these positions (m): {hotspots}")

# Track map with hotspots
try:
    circuit_info = session.get_circuit_info()
    fig_track = go.Figure()
    fig_track.add_trace(go.Scatter(x=circuit_info.corners['X'], 
                                y=circuit_info.corners['Y'],
                                mode='lines+markers',
                                name='Track'))
    for dist in hotspots:
        corner = circuit_info.corners.iloc[(circuit_info.corners['Distance']-dist).abs().argsort()[0]]
        fig_track.add_trace(go.Scatter(x=[corner['X']], y=[corner['Y']],
                            mode='markers', marker=dict(size=15, color='red'),
                            name=f'Overtaking Zone ({dist}m)'))
    st.plotly_chart(fig_track)
except:
    st.warning("Track map unavailable for this event")

# Tire analysis
tire_analysis(session)

# Weather data
st.header("🌦️ Session Weather")
st.write(session.weather_data)
