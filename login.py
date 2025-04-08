import streamlit as st
import requests

CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
CLIENT_SECRET = st.secrets["STRAVA_SECRET"]
REDIRECT_URI = "https://strava-analysis.streamlit.app/"

STRAVA_AUTH_URL = (
    f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}"
    f"&response_type=code&redirect_uri={REDIRECT_URI}"
    f"&approval_prompt=force&scope=read,activity:read_all"
)

# Define the function to generate the authorization URL
def get_authorization_url():
    """
    Generate the authorization URL that the user needs to visit in order to authenticate
    and authorize the application to access their Strava data.
    """
    auth_url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={REDIRECT_URI}&"
        f"approval_prompt=force&"
        f"scope=read,activity:read_all"
    )
    return auth_url

def login():
    """
    Function to handle login process and check if the user is authenticated.
    """
    code = st.query_params.get("code")
    if isinstance(code, list):
        code = code[0]

    if not code and "access_token" not in st.session_state:
        st.markdown(f"[**Log in with Strava**]({get_authorization_url()})", unsafe_allow_html=True)
        return False

    if code and "access_token" not in st.session_state:
        with st.spinner("Authenticating with Strava..."):
            response = requests.post("https://www.strava.com/oauth/token", data={
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': REDIRECT_URI
            })

            # Debug log
            st.write("Strava response:", response.status_code, response.text)

            if response.status_code == 200:
                tokens = response.json()
                st.session_state["access_token"] = tokens["access_token"]
                st.session_state["refresh_token"] = tokens["refresh_token"]
                st.session_state["athlete_id"] = tokens["athlete"]["id"]
                st.success("✅ Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("❌ Authentication failed.")
                return False

    return True

def get_access_token():
    """
    Retrieve the access token from session state.
    """
    return st.session_state.get("access_token")