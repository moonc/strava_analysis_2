import streamlit as st
import folium
from streamlit_folium import st_folium
import fetch
import analyze_data
import branca.colormap as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import xgboost as xgb
import numpy as np
import pandas as pd

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
if selected_activities:
    tabs = st.tabs([f"Map: {name}" for name in selected_activities] + ["Charts", "ML"])

    chart_data = []
    all_hr, all_elev, all_times, all_distances = [], [], [], []

    for idx, activity_name in enumerate(selected_activities):
        activity_id = activity_map.get(activity_name)
        access_token = fetch.get_access_token()
        keys = ['latlng', 'heartrate', 'altitude', 'time', 'distance']
        stream = fetch.get_activity_stream(activity_id, access_token, keys=keys)

        coords = stream.get('latlng', {}).get('data', [])
        heartrates = stream.get('heartrate', {}).get('data', [])
        elevations = stream.get('altitude', {}).get('data', [])
        times = stream.get('time', {}).get('data', [])
        distances = stream.get('distance', {}).get('data', [])

        all_hr.extend(heartrates)
        all_elev.extend(elevations)
        all_times.extend(times)
        all_distances.extend(distances)

        chart_data.append((activity_name, heartrates, elevations, times, distances))

    # --- ML Tab ---
    with tabs[-1]:
        st.subheader("Compare Models: Predict HR and Pace (Aggregate Data)")

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators, random_state=42),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_estimators, random_state=42),
                    }

        selected_models = {k: v for k, v in models.items() if k in selected_models}

        export_data = []
        metrics_summary = []
        pace_models = {}

        col1, col2 = st.columns(2)

        with col1:
            if all_hr and all_elev and len(all_hr) == len(all_elev):
                X_hr = np.array(all_elev).reshape(-1, 1)
                y_hr = np.array(all_hr)

                for name, model in selected_models.items():
                    model.fit(X_hr, y_hr)
                    pred = model.predict(X_hr)
                    residuals = y_hr - pred

                    r2 = model.score(X_hr, y_hr)
                    mae = mean_absolute_error(y_hr, pred)
                    rmse = np.sqrt(mean_squared_error(y_hr, pred))
                    metrics_summary.append((name, "HR", r2, mae, rmse))

                    export_data.append(pd.DataFrame({"Model": name, "Actual HR": y_hr, "Predicted HR": pred}))

        with col2:
            if all_times and all_distances and all_hr and all_elev:
                X_pace = np.column_stack((all_times, all_hr, all_elev))
                y_pace = np.array(all_distances)

                for name, model in selected_models.items():
                    model.fit(X_pace, y_pace)
                    pace_models[name] = model
                    pred = model.predict(X_pace)
                    pred_pace = np.diff(pred, prepend=0)
                    pred_pace = np.where(pred_pace <= 0, np.nan, 60 / (pred_pace / 1609.34))
                    residuals = y_pace - pred

                    r2 = model.score(X_pace, y_pace)
                    mae = mean_absolute_error(y_pace, pred)
                    rmse = np.sqrt(mean_squared_error(y_pace, pred))
                    metrics_summary.append((name, "Pace", r2, mae, rmse))

                    export_data.append(pd.DataFrame({"Model": name, "Actual Distance": y_pace, "Predicted Distance": pred}))

        if metrics_summary:
            st.markdown("### Model Comparison Summary")
            summary_df = pd.DataFrame(metrics_summary, columns=["Model", "Target", "R²", "MAE", "RMSE"])
            st.dataframe(summary_df)
            st.bar_chart(summary_df.pivot(index="Model", columns="Target", values="R²"))

        if export_data:
            export_df = pd.concat(export_data, ignore_index=True)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "model_predictions.csv", "text/csv")

        st.markdown("### Predict Average Pace for a Given Distance")
        selected_model_name = st.selectbox("Choose Model for Prediction", list(pace_models.keys()))
        input_distance = st.number_input("Enter Distance (in meters)", min_value=100.0, step=100.0)
        if input_distance > 0:
            model = pace_models[selected_model_name]
            input_features = np.array([[max(all_times), np.mean(all_hr), np.mean(all_elev)]])
            predicted_time = model.predict(input_features)[0]
            avg_pace = (predicted_time / 60) / (input_distance / 1609.34)  # min/mile
            st.success(f"Predicted Average Pace: {avg_pace:.2f} min/mile")
else:
    st.info("Please select one or two activities for comparison.")
