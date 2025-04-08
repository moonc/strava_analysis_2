import webbrowser
import requests
from flask import Flask, request

# CONFIGURATION
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
redirect_uri = 'http://localhost:5000/callback'
scopes = 'read,activity:read'

# Start Flask app
app = Flask(__name__)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    print(f'\nReceived code: {code}')

    # Exchange code for tokens
    token_url = 'https://www.strava.com/oauth/token'
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'grant_type': 'authorization_code'
    }
    res = requests.post(token_url, data=payload)
    tokens = res.json()

    print('\n✅ Tokens received:')
    print(f"Access Token:  {tokens.get('access_token')}")
    print(f"Refresh Token: {tokens.get('refresh_token')}")
    print(f"Expires At:    {tokens.get('expires_at')}")
    
    return "<h1>All set! You can close this window.</h1>"

if __name__ == '__main__':
    # Launch browser for user login
    auth_url = (
        f"https://www.strava.com/oauth/authorize?client_id={client_id}"
        f"&response_type=code&redirect_uri={redirect_uri}"
        f"&approval_prompt=force&scope={scopes}"
    )
    print("Opening browser for authorization...")
    webbrowser.open(auth_url)

    # Run the local server to capture the redirect
    app.run(port=5000)
