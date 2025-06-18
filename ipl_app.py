import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- TEAM LOGO MAPPING ---
team_logos = {
    'Sunrisers Hyderabad': 'https://i.imgur.com/kNjX8Xz.png',
    'Mumbai Indians': 'https://i.imgur.com/hzT4D2O.png',
    'Royal Challengers Bangalore': 'https://i.imgur.com/VcG8AfH.png',
    'Kolkata Knight Riders': 'https://i.imgur.com/og5RzEH.png',
    'Kings XI Punjab': 'https://i.imgur.com/9u0IqBG.png',
    'Chennai Super Kings': 'https://i.imgur.com/4p6Un7d.png',
    'Rajasthan Royals': 'https://i.imgur.com/XBRdhbC.png',
    'Delhi Capitals': 'https://i.imgur.com/6zP4NML.png'
}

teams = list(team_logos.keys())

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# --- PAGE CONFIG ---
st.set_page_config(page_title="IPL Win Predictor", layout="wide")
st.title('🏏 IPL Win Predictor')
st.markdown("### Predict the match outcome based on live match stats")

# --- TEAM SELECTION ---
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
    st.image(team_logos[batting_team], width=120, caption=batting_team)

with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))
    st.image(team_logos[bowling_team], width=120, caption=bowling_team)

if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling teams must be different.")

# --- MATCH INPUTS ---
selected_city = st.selectbox('Select Host City', sorted(cities))
target = st.number_input('Target Score', min_value=0, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# --- PREDICTION ---
if st.button('Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("Batting and bowling teams cannot be the same.")
    else:
        # Load model
        if not os.path.exists("mdl.pkl"):
            st.error("❌ Model file 'mdl.pkl' not found. Please place it in the same directory.")
            st.stop()

        with open('mdl.pkl', 'rb') as file:
            pipe = pickle.load(file)

        # Compute match stats
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

        try:
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            # Pie Chart
            fig, ax = plt.subplots()
            ax.pie([win, loss], labels=[batting_team, bowling_team], autopct='%1.1f%%',
                   colors=["#4CAF50", "#FF5722"])
            ax.set_title('📊 Win Probability')
            st.pyplot(fig)

            # Display results
            st.success(f"🏆 {batting_team} has a {round(win * 100)}% chance to win!")
            st.info(f"💀 {bowling_team} has a {round(loss * 100)}% chance to win.")

            # Match stats summary
            st.markdown("### 📈 Match Stats")
            st.write(f"🎯 Runs Left: `{runs_left}`")
            st.write(f"🕒 Balls Left: `{balls_left}`")
            st.write(f"🧱 Wickets Remaining: `{wickets_remaining}`")
            st.write(f"⚡ Current Run Rate (CRR): `{crr:.2f}`")
            st.write(f"🔥 Required Run Rate (RRR): `{rrr:.2f}`")

        except Exception as e:
            st.error(f"Prediction error: {e}")

