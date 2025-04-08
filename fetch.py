import requests
import pandas as pd
import streamlit as st
import time

# Caching the fetching of activity data
@st.cache_data(ttl=3600, show_spinner=True)  # Cache for 1 hour (3600 seconds)
def fetch_activities(access_token, per_page=100, max_pages=1):
    all_activities = []
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # First, check the rate limit before attempting any fetching
    if not check_rate_limit(headers):
        return []  # If rate limit exceeded, do not fetch activities

    for page in range(1, max_pages + 1):
        url = 'https://www.strava.com/api/v3/athlete/activities'
        params = {'per_page': per_page, 'page': page}
        
        res = requests.get(url, headers=headers, params=params)

        # If rate limit exceeded during fetching, handle it
        if res.status_code == 403:  # Rate limit exceeded (403 status)
            st.warning("Rate limit exceeded during fetching. Retrying after wait.")
            handle_rate_limit(res.headers)  # Handle the rate limit and retry
            res = requests.get(url, headers=headers, params=params)  # Retry the request after waiting
        
        if res.status_code != 200:
            raise Exception("Failed to fetch activities:", res.text)

        activities = res.json()
        if not activities:
            break

        all_activities.extend(activities)

    return all_activities


def handle_rate_limit(headers):
    remaining_requests = headers.get("X-RateLimit-Remaining")
    if remaining_requests == "0":
        reset_time = int(headers.get("X-RateLimit-Reset"))
        wait_time = reset_time - time.time()  # Calculate wait time in seconds
        st.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")  # Show the warning message
        time.sleep(wait_time)  # Wait until the rate limit resets

def check_rate_limit(headers):
    url = "https://www.strava.com/api/v3/athletes/self"  # Simple endpoint to get rate limit info
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        raise Exception("Failed to check rate limit:", res.text)
    
    remaining_requests = res.headers.get("X-RateLimit-Remaining")
    if remaining_requests == "0":
        reset_time = int(res.headers.get("X-RateLimit-Reset"))
        wait_time = reset_time - time.time()  # Calculate wait time in seconds
        st.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds before fetching activities.")
        time.sleep(wait_time)  # Wait until the rate limit resets
        return False  # Do not proceed with fetching activities yet

    return True  # Proceed with fetching activities if rate limit allows


def get_activity_detail(activity_id, access_token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {'Authorization': f'Bearer {access_token}'}
    res = requests.get(url, headers=headers)

    # Handle rate limit before fetching the activity detail
    if res.status_code == 403:
        st.warning("Rate limit exceeded during fetching activity details. Retrying after wait.")
        handle_rate_limit(res.headers)
        res = requests.get(url, headers=headers)  # Retry after waiting

    if res.status_code != 200:
        raise Exception("Failed to get activity details:", res.text)

    return res.json()


@st.cache_data(ttl=3600, show_spinner=True)  # Cache for 1 hour (3600 seconds)
def get_activity_stream(activity_id, access_token, keys=['latlng']):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'keys': ','.join(keys), 'key_by_type': 'true'}

    res = requests.get(url, headers=headers, params=params)

    # Handle rate limit before fetching the activity stream
    if res.status_code == 403:
        st.warning("Rate limit exceeded during fetching activity stream. Retrying after wait.")
        handle_rate_limit(res.headers)
        res = requests.get(url, headers=headers, params=params)  # Retry after waiting

    if res.status_code != 200:
        raise Exception("Failed to get activity stream:", res.text)

    return res.json()


def acquire_data(access_token):
    activities = fetch_activities(access_token, per_page=100, max_pages=1)  # ~200 activities
    if not activities:
        print("No activities found.")
        return
    return pd.DataFrame(activities)
