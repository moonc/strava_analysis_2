import requests
import pandas as pd
import streamlit as st

def fetch_activities(access_token, per_page=100, max_pages=2):
    all_activities = []
    headers = {'Authorization': f'Bearer {access_token}'}

    for page in range(1, max_pages + 1):
        url = 'https://www.strava.com/api/v3/athlete/activities'
        params = {'per_page': per_page, 'page': page}
        res = requests.get(url, headers=headers, params=params)

        if res.status_code != 200:
            raise Exception("Failed to fetch activities:", res.text)

        activities = res.json()
        if not activities:
            break

        all_activities.extend(activities)

    return all_activities


def get_activity_detail(activity_id, access_token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {'Authorization': f'Bearer {access_token}'}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise Exception("Failed to get activity details:", res.text)
    return res.json()

def get_activity_stream(activity_id, access_token, keys=['latlng']):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'keys': ','.join(keys), 'key_by_type': 'true'}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise Exception("Failed to get activity stream:", res.text)
    return res.json()

def acquire_data(access_token):
    activities = fetch_activities(access_token, per_page=100, max_pages=2)  # ~200 activities
    if not activities:
        print("No activities found.")
        return
    return pd.DataFrame(activities)

