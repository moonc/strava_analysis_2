import streamlit as st
import requests

CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
CLIENT_SECRET = st.secrets["STRAVA_SECRET"]
REDIRECT_URI = "https://stravaanalysis2-lhuvhrc7wna8bbdv8e4kr9.streamlit.app/"
#REDIRECT_URI = "http://localhost:8501/"

STRAVA_AUTH_URL = (
    f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}"
    f"&response_type=code&redirect_uri={REDIRECT_URI}"
    f"&approval_prompt=force&scope=read,activity:read_all"
)

def login():
    code = st.query_params.get("code")
    if isinstance(code, list):
        code = code[0]


    if not code and "access_token" not in st.session_state:
        st.markdown(f"[**Log in with Strava**]({STRAVA_AUTH_URL})", unsafe_allow_html=True)
        return False

    if code and "access_token" not in st.session_state:
        with st.spinner("Authenticating with Strava..."):
            response = requests.post("https://www.strava.com/oauth/token", data={
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'code': code,
                'grant_type': 'authorization_code'
            })

            # Debug log
            st.write("Strava response:", response.status_code, response.text)

            if response.status_code == 200:
                tokens = response.json()
                st.session_state["access_token"] = tokens["access_token"]
                st.session_state["refresh_token"] = tokens["refresh_token"]
                st.session_state["athlete_id"] = tokens["athlete"]["id"]
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Authentication failed.")
                return False

    return True

def get_access_token():
    return st.session_state.get("access_token")
