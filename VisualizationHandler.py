import plotly.graph_objects as go
import fetch
import streamlit as st

class VisualizationHandler:
    def __init__(self, selected_activities, all_hr, all_elev, all_times, all_distances, activity_map, access_token):
        self.selected_activities = selected_activities
        self.all_hr = all_hr
        self.all_elev = all_elev
        self.all_times = all_times
        self.all_distances = all_distances
        self.activity_map = activity_map  # Store the activity_map here
        self.access_token = access_token  # Store access_token
        self.chart_data = []

    def create_charts(self):
        chart_data = []
        for idx, activity_name in enumerate(self.selected_activities):
            activity_id = self.activity_map.get(activity_name)  # Use the activity_map here
            keys = ['latlng', 'heartrate', 'altitude', 'time', 'distance']
            stream = fetch.get_activity_stream(activity_id, self.access_token, keys=keys)  # Pass access_token here

            coords = stream.get('latlng', {}).get('data', [])
            heartrates = stream.get('heartrate', {}).get('data', [])
            elevations = stream.get('altitude', {}).get('data', [])
            times = stream.get('time', {}).get('data', [])
            distances = stream.get('distance', {}).get('data', [])

            self.all_hr.extend(heartrates)
            self.all_elev.extend(elevations)
            self.all_times.extend(times)
            self.all_distances.extend(distances)

            chart_data.append((activity_name, heartrates, elevations, times, distances))

        return chart_data

    def display_charts(self, chart_data):
        st.subheader("Charts")
        for name, hr, elev, time, dist in chart_data:
            st.markdown(f"#### {name}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=hr, mode='lines', name='Heart Rate', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=dist, y=elev, mode='lines', name='Elevation', yaxis='y2', line=dict(color='green')))

            fig.update_layout(
                xaxis=dict(title='Time (s)'),
                yaxis=dict(title='Heart Rate (bpm)', color='red'),
                yaxis2=dict(title='Elevation (m)', overlaying='y', side='right', color='green'),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)