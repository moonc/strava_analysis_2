import requests

client_id = '154789'
client_secret = '03aa5040cd200e41c6bd059f8a035cda1fda926a'
authorization_code = 'c5e47e572197da318a29c9f24d9ddf7bb5c6fee6'
refresh_token = '23684515e0bcd8d85171062e2919c9d0ea50dc0f'
res = requests.post(
    'https://www.strava.com/oauth/token',
    data={
        'client_id': client_id,
        'client_secret': client_secret,
        'code': authorization_code,
        'grant_type': 'authorization_code'
    }
)

tokens = res.json()
print("Access Token:", tokens['access_token'])
print("Refresh Token:", tokens['refresh_token'])  # <- Save this for future use
