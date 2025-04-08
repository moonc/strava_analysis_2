import streamlit as st
import login

class LoginHandler:
    def __init__(self):
        self.access_token = None

    def authenticate(self):
        if not login.login():
            st.stop()

        self.access_token = login.get_access_token()

        if not self.check_access_token_validity(self.access_token):
            st.stop()  # Stop execution if the access token is invalid

    def check_access_token_validity(self, access_token):
        return login.check_access_token_validity(access_token)
