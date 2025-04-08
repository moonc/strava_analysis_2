import requests
import pandas as pd

# 🔐 Your credentials
client_id = '154789'
client_secret = '03aa5040cd200e41c6bd059f8a035cda1fda926a'
refresh_token = '23684515e0bcd8d85171062e2919c9d0ea50dc0f'

# 🌐 Exchange refresh token for access token
def get_access_token():
    url = 'https://www.strava.com/oauth/token'
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }

    res = requests.post(url, data=payload)
    if res.status_code != 200:
        raise Exception("Failed to get access token:", res.text)
    return res.json()['access_token']

# 🚴‍♂️ Fetch activities using the access token
def fetch_activities(access_token, per_page=10, page=1):
    headers = {'Authorization': f'Bearer {access_token}'}
    url = 'https://www.strava.com/api/v3/athlete/activities'
    params = {
        'per_page': per_page,
        'page': page
    }

    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise Exception("Failed to fetch activities:", res.text)
    return res.json()

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
    params = {
        'keys': ','.join(keys),
        'key_by_type': 'true'
    }

    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise Exception("Failed to get activity stream:", res.text)

    return res.json()


# 🧪 Run the script
def acquire_data():
    access_token = get_access_token()
    activities = fetch_activities(access_token)

    if not activities:
        print("No activities found.")
        return

    # Convert to DataFrame and show key fields
    df = pd.DataFrame(activities)
    #print(df[['id']])  # Show the first few rows of the DataFrame
    #print(df[['name', 'type', 'distance', 'moving_time', 'start_date']])

    return df

