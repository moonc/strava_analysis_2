import streamlit as st
import folium
from streamlit_folium import st_folium
import login
import fetch
import analyze_data
import branca.colormap as cm
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Authenticate user
if not login.login():
    st.stop()

access_token = login.get_access_token()

st.set_page_config(page_title="Strava Mini Dashboard", layout="wide")
st.title("App (Main Dashboard)")

# --- Load and filter data ---
df = analyze_data.import_data(access_token, run=True)
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

model_options = [
    "Linear Regression", 
    "Random Forest", 
    "Support Vector Regressor",
    "Gradient Boosting", 
]
selected_models = st.sidebar.multiselect("Select ML Models", model_options, default=model_options)

# --- Hyperparameters ---
st.sidebar.header("Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees (for Forest/Boosting)", 10, 300, 100, step=10)

# --- Main Display ---
mobile_mode = st.checkbox("📱 Mobile Layout", value=False)

if selected_activities:
    tabs = st.tabs([f"Map: {name}" for name in selected_activities] + ["Charts", "ML", "Performance Trends"])

    chart_data = []
    all_hr, all_elev, all_times, all_distances, all_dates = [], [], [], [], []

    for idx, activity_name in enumerate(selected_activities):
        activity_id = activity_map.get(activity_name)
        keys = ['latlng', 'heartrate', 'altitude', 'time', 'distance']
        stream = fetch.get_activity_stream(activity_id, access_token, keys=keys)

        with tabs[idx]:
            analyze_data.map_activity(activity_id, access_token, key=f"map_{activity_id}", color_by=color_by)

        coords = stream.get('latlng', {}).get('data', [])
        heartrates = stream.get('heartrate', {}).get('data', [])
        elevations = stream.get('altitude', {}).get('data', [])
        times = stream.get('time', {}).get('data', [])
        distances = stream.get('distance', {}).get('data', [])
        dates = [activity['start_date'] for activity in df.to_dict('records')]

        all_hr.extend(heartrates)
        all_elev.extend(elevations)
        all_times.extend(times)
        all_distances.extend(distances)
        all_dates.extend(dates)

        chart_data.append((activity_name, heartrates, elevations, times, distances, dates))

    # --- Performance Trends Tab ---
    with tabs[-1]:
        st.subheader("User's Performance Trends Over Time")

        # Convert date strings to datetime objects
        df['start_date'] = pd.to_datetime(df['start_date'])

        # Plot Heart Rate and Pace Trends
        st.subheader("Heart Rate and Pace Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['start_date'], y=df['heartrate'], mode='lines', name='Heart Rate', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['start_date'], y=df['pace'], mode='lines', name='Pace', line=dict(color='blue')))
        fig.update_layout(title="Heart Rate and Pace Trends Over Time", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

        # Plot Elevation Gain Over Time
        st.subheader("Elevation Gain Over Time")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['start_date'], y=df['elevation'], mode='lines', name='Elevation Gain', line=dict(color='green')))
        fig2.update_layout(title="Elevation Gain Over Time", xaxis_title="Date", yaxis_title="Elevation (m)")
        st.plotly_chart(fig2, use_container_width=True)

        # Weekly/Monthly Totals
        st.subheader("Weekly/Monthly Totals")
        df['week'] = df['start_date'].dt.strftime('%Y-%U')  # Week number
        df['month'] = df['start_date'].dt.strftime('%Y-%m')  # Month
        weekly_totals = df.groupby('week').agg({'distance': 'sum', 'time': 'sum', 'elevation': 'sum'}).reset_index()
        monthly_totals = df.groupby('month').agg({'distance': 'sum', 'time': 'sum', 'elevation': 'sum'}).reset_index()

        # Plot Weekly Totals
        st.subheader("Weekly Totals")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=weekly_totals['week'], y=weekly_totals['distance'], name="Distance (m)", marker_color='blue'))
        fig3.add_trace(go.Bar(x=weekly_totals['week'], y=weekly_totals['time'], name="Time (s)", marker_color='orange'))
        fig3.update_layout(title="Weekly Totals", xaxis_title="Week", yaxis_title="Total")
        st.plotly_chart(fig3, use_container_width=True)

        # Plot Monthly Totals
        st.subheader("Monthly Totals")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=monthly_totals['month'], y=monthly_totals['distance'], name="Distance (m)", marker_color='blue'))
        fig4.add_trace(go.Bar(x=monthly_totals['month'], y=monthly_totals['time'], name="Time (s)", marker_color='orange'))
        fig4.update_layout(title="Monthly Totals", xaxis_title="Month", yaxis_title="Total")
        st.plotly_chart(fig4, use_container_width=True)

    # --- ML Tab ---
    with tabs[-2]:
        st.subheader("Compare Models: Predict HR and Pace (Aggregate Data)")

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators, random_state=42),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_estimators, random_state=42),
        }

        selected_models_dict = {k: v for k, v in models.items() if k in selected_models}
        export_data = []
        metrics_summary = []
        pace_models = {}

        col1, col2 = st.columns(2)

        with col1:
            if all_hr and all_elev and len(all_hr) == len(all_elev):
                X_hr = np.array(all_elev).reshape(-1, 1)
                y_hr = np.array(all_hr)
                for name, model in selected_models_dict.items():
                    model.fit(X_hr, y_hr)
                    pred = model.predict(X_hr)
                    r2 = model.score(X_hr, y_hr)
                    mae = mean_absolute_error(y_hr, pred)
                    rmse = np.sqrt(mean_squared_error(y_hr, pred))
                    metrics_summary.append((name, "HR", r2, mae, rmse))
                    export_data.append(pd.DataFrame({"Model": name, "Actual HR": y_hr, "Predicted HR": pred}))

        with col2:
            if all_times and all_distances and all_hr and all_elev:
                X_pace = np.column_stack((all_times, all_hr, all_elev))
                y_pace = np.array(all_distances)
                for name, model in selected_models_dict.items():
                    model.fit(X_pace, y_pace)
                    pace_models[name] = model
                    pred = model.predict(X_pace)
                    r2 = model.score(X_pace, y_pace)
                    mae = mean_absolute_error(y_pace, pred)
                    rmse = np.sqrt(mean_squared_error(y_pace, pred))
                    metrics_summary.append((name, "Pace", r2, mae, rmse))
                    export_data.append(pd.DataFrame({"Model": name, "Actual Distance": y_pace, "Predicted Distance": pred}))

        if metrics_summary:
            st.markdown("### Model Comparison Summary")
            summary_df = pd.DataFrame(metrics_summary, columns=["Model", "Target", "R²", "MAE", "RMSE"])
            st.dataframe(summary_df)

            # Display Bar Chart for R² Scores
            st.subheader("R² Comparison")
            st.bar_chart(summary_df.pivot(index="Model", columns="Target", values="R²"))

            best_models = summary_df.sort_values("R²", ascending=False).groupby("Target").first().reset_index()
            for _, row in best_models.iterrows():
                st.info(f"Best model for {row['Target']}: {row['Model']} (R²: {row['R²']:.2f}, MAE: {row['MAE']:.2f}, RMSE: {row['RMSE']:.2f})")

        if export_data:
            export_df = pd.concat(export_data, ignore_index=True)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "model_predictions.csv", "text/csv")

else:
    st.info("Please select one or two activities for comparison.")
