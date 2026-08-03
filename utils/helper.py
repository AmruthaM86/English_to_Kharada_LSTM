import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS_FILE = os.path.join(
    BASE_DIR,
    "assets",
    "css",
    "style.css"
)

def load_css():

    if os.path.exists(CSS_FILE):

        with open(CSS_FILE) as f:

            st.markdown(

                f"<style>{f.read()}</style>",

                unsafe_allow_html=True

            )