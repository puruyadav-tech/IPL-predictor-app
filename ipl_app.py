import streamlit as st
import pandas as pd
import pickle
import os

# --- BLUE THEME INLINE CSS ---
st.markdown("""
    <style>
    body {
      background: linear-gradient(to bottom right, #1e3c72, #2a5298);
      color: white;
      font-family: 'Segoe UI', sans-serif;
    }
    .title {
      background: linear-gradient(to right, #fbbf24, #f97316, #ffffff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 3.5rem;
      text-align: center;
      font-weight: 800;
      margin-top: 0.5rem;
    }
    .subtitle {
      text-align: center;
      font-size: 1.2rem;
      color: #dddddd;
      margin-bottom: 2rem;
    }
    .card {
      background-color: rgba(255, 255, 255, 0.05);
      border-radius: 20px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 0 20px rgba(0,0,0,0.2);
    }
    .result-card {
      background-color: rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 20px;
      margin-top: 30px;
      text-align: center;
      font-size: 1.4rem;
      color: #f1f1f1;
      box-shadow: 0 0 25px rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- TEAM LIST ---
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# --- PAGE HEADER ---
st.set_page_config(page_title="IPL Win Predictor", layout="wide")
st.markdown("<h1 class='title'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict the match outcome based on live match stats</p>", unsafe_allow_html=True)

# --- TEAM SELECTION ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))
st.markdown("</div>", unsafe_allow_html=True)

if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling teams must be different.")

# --- MATCH DETAILS ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
selected_city = st.selectbox('Select Host City', sorted(cities))
target = st.number_input('🎯 Target Score', min_value=0, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('📊 Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('🕒 Overs Completed', min_value=0.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('💥 Wickets Fallen', min_value=0, max_value=10, step=1)
st.markdown("</div>", unsafe_allow_html=True)

# --- PREDICT WIN ---
if st.button("🚀 Predict Match Outcome"):
    if batting_team == bowling_team:
        st.error("❌ Batting and bowling teams cannot be the same.")
    elif not os.path.exists("mdl.pkl"):
        st.error("Model file 'mdl.pkl' not found.")
    else:
        with open('mdl.pkl', 'rb') as f:
            pipe = pickle.load(f)

        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_remaining = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.markdown(
            f"<div class='result-card'><h3>🏆 {batting_team} Chance to Win: {round(win*100)}%</h3>"
            f"<h4>💀 {bowling_team} Chance: {round(loss*100)}%</h4></div>",
            unsafe_allow_html=True
        )
