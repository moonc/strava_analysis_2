import streamlit as st
import folium
from streamlit_folium import st_folium
import fetch
import branca.colormap as cm
import numpy as np
import pandas as pd

def import_data(access_token, run=False):
    df = fetch.acquire_data(access_token)
    if df is None or df.empty:
        print("No activities found.")
        return
    running_df = df[df['type'] == 'Run'].reset_index(drop=True)
    return running_df if run else df

def get_ids(df):
    return df['id'].tolist()

def get_activity_detail(activity_id, access_token, save_to_csv=False, print_info=False):
    detail = fetch.get_activity_detail(activity_id, access_token)
    df = pd.DataFrame([detail])
    if print_info:
        print(df[['name', 'type', 'distance', 'moving_time', 'start_date']])
    if save_to_csv:
        df.to_csv('activity_details.csv', index=False)
        print("Activity details saved to activity_details.csv")

def map_activity(activity_id, access_token, key=None, color_by="None"):
    stream = fetch.get_activity_stream(activity_id, access_token, keys=['latlng', 'heartrate', 'altitude'])
    if 'latlng' not in stream or not stream['latlng']['data']:
        st.warning("No GPS data available for this activity.")
        return

    gps_coords = stream['latlng']['data']
    m = folium.Map(location=gps_coords[0], zoom_start=13)

    if color_by == "Heart Rate (Gradient)" and 'heartrate' in stream:
        hr_data = stream['heartrate']['data']
        colormap = cm.linear.YlOrRd_09.scale(min(hr_data), max(hr_data))
        for i in range(1, len(gps_coords)):
            folium.PolyLine(
                [gps_coords[i-1], gps_coords[i]],
                color=colormap(hr_data[i]),
                weight=4
            ).add_to(m)
        colormap.caption = "Heart Rate (bpm)"
        colormap.add_to(m)

    elif color_by == "Heart Rate (Zones)" and 'heartrate' in stream:
        hr_data = stream['heartrate']['data']
        for i in range(1, len(gps_coords)):
            hr = hr_data[i]
            color = "green" if hr < 120 else "orange" if hr < 150 else "red"
            folium.PolyLine([gps_coords[i-1], gps_coords[i]], color=color, weight=4).add_to(m)

    elif color_by == "Elevation" and 'altitude' in stream:
        elev_data = stream['altitude']['data']
        colormap = cm.linear.PuBuGn_09.scale(min(elev_data), max(elev_data))
        for i in range(1, len(gps_coords)):
            folium.PolyLine(
                [gps_coords[i-1], gps_coords[i]],
                color=colormap(elev_data[i]),
                weight=4
            ).add_to(m)
        colormap.caption = "Elevation (m)"
        colormap.add_to(m)

    else:
        folium.PolyLine(gps_coords, color='blue', weight=4).add_to(m)

    folium.Marker(gps_coords[0], tooltip="Start").add_to(m)
    folium.Marker(gps_coords[-1], tooltip="End").add_to(m)
    st_folium(m, width=700, height=500, key=key)

    # Display basic stats
    if 'heartrate' in stream:
        hr_data = stream['heartrate']['data']
        st.markdown(f"**Heart Rate Stats:** 🫀")
        st.write(f"Average: {np.mean(hr_data):.1f} bpm")
        st.write(f"Max: {np.max(hr_data):.1f} bpm")
        st.write(f"Min: {np.min(hr_data):.1f} bpm")

    if 'altitude' in stream:
        elev_data = stream['altitude']['data']
        st.markdown(f"**Elevation Stats:** 🏔️")
        st.write(f"Gain: {np.max(elev_data) - np.min(elev_data):.1f} m")
        st.write(f"Max Elevation: {np.max(elev_data):.1f} m")
        st.write(f"Min Elevation: {np.min(elev_data):.1f} m")

    if 'time' in stream and 'distance' in stream:
        time_data = stream['time']['data']
        dist_data = stream['distance']['data']
        if time_data and dist_data and dist_data[-1] > 0:
            pace = (time_data[-1] / 60) / (dist_data[-1] / 1609.34)
            st.markdown(f"**Pace Stats:** 🏃")
            st.write(f"Total Time: {time_data[-1] / 60:.1f} min")
            st.write(f"Total Distance: {dist_data[-1] / 1609.34:.2f} mi")
            st.write(f"Avg Pace: {pace:.2f} min/mile")
