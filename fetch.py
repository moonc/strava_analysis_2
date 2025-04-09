import requests
import time
import streamlit as st
import pandas as pd

CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
CLIENT_SECRET = st.secrets["STRAVA_SECRET"]

# Cache the fetching of activity data
@st.cache_data(ttl=3600, show_spinner=True)  # Cache for 1 hour (3600 seconds)
def fetch_activities(access_token, per_page=100, max_pages=1):
    all_activities = []
    headers = {'Authorization': f'Bearer {access_token}'}

    for page in range(1, max_pages + 1):
        url = 'https://www.strava.com/api/v3/athlete/activities'
        params = {'per_page': per_page, 'page': page}

        # Check rate limit before making the API call
        if check_rate_limit(headers):
            res = requests.get(url, headers=headers, params=params)

            if res.status_code != 200:
                raise Exception("Failed to fetch activities:", res.text)

            activities = res.json()
            if not activities:
                break

            all_activities.extend(activities)
        else:
            st.warning("Rate limit reached. Waiting for reset...")

    return all_activities


def check_rate_limit(headers):
    """Check the Strava API rate limit status and wait if necessary."""
    url = "https://www.strava.com/api/v3/athlete/activities"
    res = requests.get(url, headers=headers)

    remaining_requests = res.headers.get("X-RateLimit-Remaining")
    if remaining_requests == "0":
        reset_time = int(res.headers.get("X-RateLimit-Reset"))
        wait_time = reset_time - time.time()
        if wait_time > 0:
            st.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)  # Sleep until the rate limit resets
            return True
        else:
            st.error("Failed to wait for rate limit reset.")
            return False

    return True


def get_activity_detail(activity_id, access_token):
    """Fetch detailed activity data."""
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {'Authorization': f'Bearer {access_token}'}

    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise Exception("Failed to get activity details:", res.text)

    return res.json()


@st.cache_data(ttl=3600, show_spinner=True)  # Cache for 1 hour (3600 seconds)
def get_activity_stream(activity_id, access_token, keys=['latlng']):
    """Fetch activity streams."""
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'keys': ','.join(keys), 'key_by_type': 'true'}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise Exception("Failed to get activity stream:", res.text)
    return res.json()


def acquire_data(access_token):
    """Fetch activity data and store in a DataFrame."""
    activities = fetch_activities(access_token, per_page=100, max_pages=1)  # ~200 activities
    if not activities:
        print("No activities found.")
        return
    return pd.DataFrame(activities)
