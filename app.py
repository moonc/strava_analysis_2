import streamlit as st
import folium
from streamlit_folium import st_folium
import fetch
import analyze_data
import branca.colormap as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Strava Mini Dashboard", layout="wide")
st.title("Strava Mini Dashboard")

# --- Load and filter data ---
df = analyze_data.import_data(run=True)  # Only running activities
activity_ids = analyze_data.get_ids(df) if df is not None else []
activity_names = df['name'].tolist() if df is not None else []
activity_map = dict(zip(activity_names, activity_ids))

# --- Sidebar controls ---
st.sidebar.header("Activity Settings")
selected_activities = st.sidebar.multiselect("Select Activities for Comparison", activity_names, max_selections=2)

color_by = st.sidebar.selectbox(
    "Color Path By:",
    ("None", "Heart Rate (Gradient)", "Heart Rate (Zones)", "Elevation")
)

# --- Main Display ---
if selected_activities:
    tabs = st.tabs([f"Map: {name}" for name in selected_activities] + ["Charts", "ML"])

    chart_data = []

    for idx, activity_name in enumerate(selected_activities):
        activity_id = activity_map.get(activity_name)
        access_token = fetch.get_access_token()
        keys = ['latlng', 'heartrate', 'altitude']
        stream = fetch.get_activity_stream(activity_id, access_token, keys=keys)

        coords = stream.get('latlng', {}).get('data', [])
        heartrates = stream.get('heartrate', {}).get('data', [])
        elevations = stream.get('altitude', {}).get('data', [])

        detail = fetch.get_activity_detail(activity_id, access_token)
        moving_time = detail.get('moving_time', 0)
        distance = detail.get('distance', 0)

        with tabs[idx]:
            if not coords:
                st.error(f"No GPS data for {activity_name}.")
                continue

            m = folium.Map(location=coords[0], zoom_start=13)

            if color_by == "Heart Rate (Gradient)" and heartrates:
                min_hr, max_hr = min(heartrates), max(heartrates)
                colormap = cm.linear.RdYlGn_11.scale(min_hr, max_hr).to_step(10)
                colormap.caption = 'Heart Rate (bpm)'

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

            st.subheader(activity_name)
            st_data = st_folium(m, width=700, height=500)

            st.markdown("**Stats:**")
            if heartrates:
                st.metric("Avg HR", f"{sum(heartrates)//len(heartrates)} bpm")
                st.metric("Max HR", f"{max(heartrates)} bpm")
                st.metric("Min HR", f"{min(heartrates)} bpm")
            else:
                st.write("No heart rate data available.")

            if elevations:
                st.metric("Max Elevation", f"{max(elevations):.1f} m")
                st.metric("Min Elevation", f"{min(elevations):.1f} m")
                st.metric("Elevation Gain", f"{elevations[-1] - elevations[0]:.1f} m")
            else:
                st.write("No elevation data available.")

            if moving_time > 0 and distance > 0:
                pace = (moving_time / 60) / (distance / 1000)  # min/km
                st.metric("Avg Pace", f"{pace:.2f} min/km")
            else:
                st.write("Pace data unavailable.")

        chart_data.append((activity_name, heartrates, elevations))

    # --- Chart Tab ---
    with tabs[-2]:
        st.subheader("Heart Rate and Elevation Charts")
        for name, hr, elev in chart_data:
            st.markdown(f"### {name}")
            col1, col2 = st.columns(2)

            with col1:
                if hr:
                    fig, ax = plt.subplots()
                    ax.plot(hr, color='red')
                    ax.set_title("Heart Rate")
                    ax.set_xlabel("Data Point")
                    ax.set_ylabel("bpm")
                    st.pyplot(fig)
                else:
                    st.write("No heart rate data available.")

            with col2:
                if elev:
                    fig, ax = plt.subplots()
                    ax.plot(elev, color='purple')
                    ax.set_title("Elevation")
                    ax.set_xlabel("Data Point")
                    ax.set_ylabel("Meters")
                    st.pyplot(fig)
                else:
                    st.write("No elevation data available.")

    # --- ML Tab ---
    with tabs[-1]:
        st.subheader("ML Experiment: Predict HR from Elevation")
        for name, hr, elev in chart_data:
            st.markdown(f"### {name}")
            if hr and elev and len(hr) == len(elev):
                X = np.array(elev).reshape(-1, 1)
                y = np.array(hr)
                model = LinearRegression().fit(X, y)
                pred_hr = model.predict(X)

                fig, ax = plt.subplots()
                ax.plot(hr, label="Actual HR", color='red')
                ax.plot(pred_hr, label="Predicted HR", linestyle='--', color='green')
                ax.set_title("Heart Rate Prediction from Elevation")
                ax.set_xlabel("Data Point")
                ax.set_ylabel("bpm")
                ax.legend()
                st.pyplot(fig)
                st.success(f"R² Score: {model.score(X, y):.2f}")
            else:
                st.warning("Insufficient or mismatched data for ML model.")
else:
    st.info("Please select one or two activities for comparison.")
