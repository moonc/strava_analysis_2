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

# Set page layout and title
st.set_page_config(page_title="Strava Mini Dashboard", layout="centered")
st.title("Strava Mini Dashboard 🏃‍♂️🚴‍♀️")
st.markdown(
    """
    Welcome to the **Strava Mini Dashboard**! 
    This app helps you analyze your Strava activities and visualize important statistics like **Heart Rate**, **Pace**, and **Elevation**.
    Please log in to get started.
    """
)

# Display a friendly welcome image (replace with a useful image)
st.image("https://www.example.com/your-image.jpg", width=300)  # Replace with your desired image URL

# Create a nice-looking login button and a function to handle login
login_button = st.button("Log in with Strava")

if login_button:
    # Redirect the user to the Strava authentication page
    authorization_url = login.get_authorization_url()  # This function should return the URL
    st.markdown(f'<a href="{authorization_url}" target="_self"><button style="background-color:#f7a80d;color:white;font-size:18px;padding:15px 25px;border-radius:10px;width:100%;font-weight:bold;">Click to Log In with Strava</button></a>', unsafe_allow_html=True)

# After the user is logged in, fetch access token
if login.login():  # Check if the user is authenticated
    access_token = login.get_access_token()

    if not fetch.check_access_token_validity(access_token):
        st.stop()  # Stop execution if the access token is invalid
    
    # Welcome the user and continue with the app
    st.success("✅ Logged in successfully! Welcome to your dashboard!")

    # Proceed with loading and displaying activity data
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
        tabs = st.tabs([f"Map: {name}" for name in selected_activities] + ["Charts", "ML"])

        chart_data = []
        all_hr, all_elev, all_times, all_distances = [], [], [], []

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

            all_hr.extend(heartrates)
            all_elev.extend(elevations)
            all_times.extend(times)
            all_distances.extend(distances)

            chart_data.append((activity_name, heartrates, elevations, times, distances))

        # --- Charts Tab ---
        with tabs[-2]:
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

        # --- ML Tab ---
        with tabs[-1]:
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
                st.bar_chart(summary_df.pivot(index="Model", columns="Target", values="R²"))
                best_models = summary_df.sort_values("R²", ascending=False).groupby("Target").first().reset_index()
                for _, row in best_models.iterrows():
                    st.info(f"Best model for {row['Target']}: {row['Model']} (R²: {row['R²']:.2f}, MAE: {row['MAE']:.2f}, RMSE: {row['RMSE']:.2f})")

            if export_data:
                export_df = pd.concat(export_data, ignore_index=True)
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "model_predictions.csv", "text/csv")

            st.markdown("### Predict Total Time and Average Pace for a Given Distance")
            selected_model_name = st.selectbox("Choose Model for Prediction", list(pace_models.keys()))
            input_unit = st.radio("Select Distance Unit", ["Kilometers", "Miles"], horizontal=True)
            prediction_type = st.radio("Prediction Type", ["Total Summary", "Segment Splits"])
            input_distance = st.number_input("Enter Distance", min_value=0.1, step=0.1, format="%.2f")
            custom_hr = st.number_input("Custom Average Heart Rate (bpm)", min_value=60, max_value=220, value=int(np.mean(all_hr)))
            custom_elev = st.number_input("Custom Average Elevation (m)", min_value=0, max_value=1000, value=int(np.mean(all_elev)))

            if input_distance > 0:
                model = pace_models[selected_model_name]
                max_time = max(all_times)
                if prediction_type == "Total Summary":
                    elevation_gain = (custom_elev / max(all_distances)) * (input_distance * 1609.34 if input_unit == "Miles" else input_distance * 1000)
                    input_features = np.array([[max_time, custom_hr, elevation_gain]])
                    predicted_total_time = model.predict(input_features)[0]
                    lower_bound = predicted_total_time * 0.95
                    upper_bound = predicted_total_time * 1.05
                    pace = (predicted_total_time / 60) / input_distance
                    unit_label = "min/mile" if input_unit == "Miles" else "min/km"
                    st.success(f"Predicted Time: {predicted_total_time/60:.2f} minutes")
                    st.info(f"Estimated Range: {lower_bound/60:.2f} - {upper_bound/60:.2f} minutes")
                    st.success(f"Predicted Average Pace: {pace:.2f} {unit_label}")
                elif prediction_type == "Segment Splits":
                    unit_distance = 1.0
                    num_segments = int(input_distance / unit_distance)
                    split_times = []
                    for _ in range(num_segments):
                        elev_gain = (custom_elev / max(all_distances)) * (unit_distance * 1609.34 if input_unit == "Miles" else unit_distance * 1000)
                        input_feat = np.array([[max_time, custom_hr, elev_gain]])
                        segment_time = model.predict(input_feat)[0] / num_segments
                        split_times.append(segment_time / 60)
                    st.markdown("**Segment Split Times:**")
                    for i, t in enumerate(split_times):
                        st.write(f"Segment {i+1}: {t:.2f} min")

else:
    st.info("Please select one or two activities for comparison.")
