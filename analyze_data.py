import fetch
import pandas as pd
import folium
from streamlit_folium import st_folium

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

def map_activity(activity_id, access_token, key=None):
    stream = fetch.get_activity_stream(activity_id, access_token)
    if 'latlng' not in stream or not stream['latlng']['data']:
        print("No GPS data available for this activity.")
        return
    gps_coords = stream['latlng']['data']
    start_latlng = gps_coords[0]
    m = folium.Map(location=start_latlng, zoom_start=13)
    folium.PolyLine(gps_coords, color='blue', weight=4).add_to(m)
    folium.Marker(gps_coords[0], tooltip="Start").add_to(m)
    folium.Marker(gps_coords[-1], tooltip="End").add_to(m)
    st_folium(m, width=700, height=500, key=key)
