import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

class MLModelHandler:
    def __init__(self, selected_models, n_estimators):
        self.selected_models = selected_models
        self.n_estimators = n_estimators
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators, random_state=42),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_estimators, random_state=42),
        }
        self.metrics_summary = []
        self.export_data = []
        self.pace_models = {}

    def train_and_evaluate(self, all_hr, all_elev, all_times, all_distances):
        selected_models_dict = {k: v for k, v in self.models.items() if k in self.selected_models}

        # Evaluate HR Prediction
        if all_hr and all_elev and len(all_hr) == len(all_elev):
            X_hr = np.array(all_elev).reshape(-1, 1)
            y_hr = np.array(all_hr)
            for name, model in selected_models_dict.items():
                model.fit(X_hr, y_hr)
                pred = model.predict(X_hr)
                r2 = model.score(X_hr, y_hr)
                mae = mean_absolute_error(y_hr, pred)
                rmse = np.sqrt(mean_squared_error(y_hr, pred))
                self.metrics_summary.append((name, "HR", r2, mae, rmse))
                self.export_data.append(pd.DataFrame({"Model": name, "Actual HR": y_hr, "Predicted HR": pred}))

        # Evaluate Pace Prediction
        if all_times and all_distances and all_hr and all_elev:
            X_pace = np.column_stack((all_times, all_hr, all_elev))
            y_pace = np.array(all_distances)
            for name, model in selected_models_dict.items():
                model.fit(X_pace, y_pace)
                self.pace_models[name] = model
                pred = model.predict(X_pace)
                r2 = model.score(X_pace, y_pace)
                mae = mean_absolute_error(y_pace, pred)
                rmse = np.sqrt(mean_squared_error(y_pace, pred))
                self.metrics_summary.append((name, "Pace", r2, mae, rmse))
                self.export_data.append(pd.DataFrame({"Model": name, "Actual Distance": y_pace, "Predicted Distance": pred}))

    def get_metrics_summary(self):
        return self.metrics_summary

    def get_export_data(self):
        return self.export_data

    def get_pace_models(self):
        return self.pace_models
