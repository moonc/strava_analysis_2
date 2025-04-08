import streamlit as st
import folium
from streamlit_folium import st_folium
import fetch
import analyze_data
import branca.colormap as cm

st.set_page_config(page_title="Strava Mini Dashboard", layout="wide")
st.title("🏃 Strava Mini Dashboard")

# --- Load and filter data ---
df = analyze_data.import_data(run=True)  # Only running activities
activity_ids = analyze_data.get_ids(df) if df is not None else []
activity_names = df['name'].tolist() if df is not None else []
activity_map = dict(zip(activity_names, activity_ids))

# --- Sidebar controls ---
st.sidebar.header("Activity Settings")
activity_name = st.sidebar.selectbox("Select an Activity", activity_names)
activity_id = activity_map.get(activity_name)

color_by = st.sidebar.selectbox(
    "Color Path By:",
    ("None", "Heart Rate (Gradient)", "Heart Rate (Zones)", "Elevation")
)

# --- Map display ---
if activity_id:
    access_token = fetch.get_access_token()
    keys = ['latlng']
    if "Heart Rate" in color_by:
        keys.append('heartrate')
    elif color_by == "Elevation":
        keys.append('altitude')

    stream = fetch.get_activity_stream(activity_id, access_token, keys=keys)

    coords = stream.get('latlng', {}).get('data', [])
    heartrates = stream.get('heartrate', {}).get('data', [])
    elevations = stream.get('altitude', {}).get('data', [])

    if not coords:
        st.error("No GPS data found for this activity.")
    else:
        m = folium.Map(location=coords[0], zoom_start=13)

        if color_by == "Heart Rate (Gradient)" and heartrates:
            min_hr, max_hr = min(heartrates), max(heartrates)
            colormap = cm.linear.RdYlGn_11.scale(min_hr, max_hr)
            colormap.caption = 'Heart Rate (bpm)'

        # Color-coded path
        for i in range(1, len(coords)):
            color = 'blue'
            tooltip = None

            if color_by == "Heart Rate (Gradient)" and heartrates:
                color = colormap(heartrates[i])
                tooltip = f"HR: {heartrates[i]} bpm"
            elif color_by == "Heart Rate (Zones)" and heartrates:
                hr = heartrates[i]
                if hr < 120:
                    color = 'green'
                elif hr < 150:
                    color = 'orange'
                else:
                    color = 'red'
                tooltip = f"HR: {hr} bpm"
            elif color_by == "Elevation" and elevations:
                elev_change = elevations[i] - elevations[i - 1]
                color = 'orange' if elev_change > 0 else 'purple'
                tooltip = f"Elevation: {elevations[i]:.1f} m"

            folium.PolyLine([coords[i - 1], coords[i]], color=color, weight=4, tooltip=tooltip).add_to(m)

        if color_by == "Heart Rate (Gradient)" and heartrates:
            colormap.add_to(m)

        col1, col2 = st.columns([2, 1])
        with col1:
            st_data = st_folium(m, width=800, height=600)

        with col2:
            st.subheader("Stats")
            if heartrates:
                st.metric("Avg HR", f"{sum(heartrates)//len(heartrates)} bpm")
                st.metric("Max HR", f"{max(heartrates)} bpm")
                st.metric("Min HR", f"{min(heartrates)} bpm")
            if elevations:
                st.metric("Max Elevation", f"{max(elevations):.1f} m")
                st.metric("Min Elevation", f"{min(elevations):.1f} m")
                st.metric("Elevation Gain", f"{elevations[-1] - elevations[0]:.1f} m")
else:
    st.info("Please select an activity to display the route.")
