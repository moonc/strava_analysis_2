import fetch
import pandas as pd
import folium

def import_data(run=False):
    df = fetch.acquire_data()
    running_df = df[df['type'] == 'Run'].reset_index(drop=True)

    if df is None or df.empty:
        print("No activities found.")
        return
    
    if run: return running_df

    return df

def get_ids(df):
    # Extract activity IDs from the DataFrame
    list_of_ids = df['id'].tolist()
    return list_of_ids

def get_activity_detail(activity_id,save_to_csv=False,print_info =False):
    access_token = fetch.get_access_token()
    activity_detail = fetch.get_activity_detail(activity_id, access_token)
    
    # Convert to DataFrame and show key fields
    df = pd.DataFrame([activity_detail])
    if print_info: print(df[['name', 'type', 'distance', 'moving_time', 'start_date']])

    if save_to_csv:# Save to CSV
        df.to_csv('activity_details.csv', index=False)
        print("Activity details saved to activity_details.csv")


def map_activity(activity_id, output_file='activity_map.html'):
    access_token = fetch.get_access_token()
    stream = fetch.get_activity_stream(activity_id, access_token)

    if 'latlng' not in stream or not stream['latlng']['data']:
        print("No GPS data available for this activity.")
        return

    gps_coords = stream['latlng']['data']

    # Center map on first point
    start_latlng = gps_coords[0]
    m = folium.Map(location=start_latlng, zoom_start=13)

    # Draw route
    folium.PolyLine(gps_coords, color='blue', weight=4).add_to(m)

    # Mark start and end
    folium.Marker(gps_coords[0], tooltip="Start").add_to(m)
    folium.Marker(gps_coords[-1], tooltip="End").add_to(m)

    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")