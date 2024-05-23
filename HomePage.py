import json
import subprocess
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras import switch_page_button

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

# Add a button to execute another Python file when clicked
if st.button("Let's find the best for you!!"):
    # Specify the path to the Python file you want to run
    another_file_path = "https://github.com/Mayuresh2703/Meal_Recommendation_System/blob/main/Meal_Recommender.py"
    
    # Use subprocess to run the streamlit command
    # subprocess.run(["streamlit", "run", another_file_path])
    switch_page_button("Switch to another page", another_file_path)

