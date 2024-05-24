import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

st.title('MEAL RECOMMENDATION SYSTEM')

# Load CSS from a file
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def get(path: str):
    with open(path, "r") as p:
        return json.load(p)

path = get("./ani.json")

# Use the full_width property to make the Lottie animation occupy the entire horizontal space
st_lottie(path, width=None)

# Use CSS to make the Lottie animation occupy the entire vertical space
st.markdown("""
    <style>
        div.stLottie {
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Add a button to switch to the meal_recommender page
if st.button("Let's find the best for you!!"):
    switch_page("Meal_Recommender")

