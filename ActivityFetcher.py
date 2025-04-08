import analyze_data
import pandas as pd

class ActivityFetcher:
    def __init__(self, access_token):
        self.access_token = access_token
        self.df = None
        self.activity_ids = []
        self.activity_names = []
        self.activity_map = {}

    def load_data(self):
        self.df = analyze_data.import_data(self.access_token, run=True)
        self.activity_ids = analyze_data.get_ids(self.df) if self.df is not None else []
        self.activity_names = self.df['name'].tolist() if self.df is not None else []
        self.activity_map = dict(zip(self.activity_names, self.activity_ids))

    def get_activity_names(self):
        return self.activity_names

    def get_activity_map(self):
        return self.activity_map
