import streamlit as st
import pandas as pd
import os
from joblib import load
import matplotlib.pyplot as plt

# --- Custom CSS for IPL Blue Theme ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
            color: white;
        }
        h1, h2, h3 {
            color: orange !important;
            text-align: center;
        }
        .css-1v0mbdj, .stButton>button {
            background-color: #1f2c3b;
            color: white;
            border: 1px solid #ff9800;
            border-radius: 12px;
            padding: 0.5rem 1rem;
        }
        .stNumberInput>div>input {
            background-color: #1f2c3b !important;
            color: white !important;
        }
        .stSelectbox>div>div {
            background-color: #1f2c3b !important;
            color: white !important;
        }
        .stMarkdown {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data Setup ---
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

# --- Title and Header ---
st.title('🏏 IPL Win Predictor')
st.markdown("### Predict the match outcome based on live match stats")

# --- Team Selection ---
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling teams must be different.")

# --- Match Inputs ---
selected_city = st.selectbox('Select Host City', sorted(cities))
target = st.number_input('🎯 Target Score', min_value=0, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('📊 Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('🕒 Overs Completed', min_value=0.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('💥 Wickets Fallen', min_value=0, max_value=10, step=1)

# --- Prediction ---
if st.button('🔥 Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("Batting and bowling teams cannot be the same.")
    else:
        if not os.path.exists("mdl.pkl"):
            st.error("❌ Model file 'mdl.pkl' not found.")
            st.stop()

        try:
            pipe = load('mdl.pkl')
        except Exception as e:
            st.error(f"❗ Model loading error: {e}")
            st.stop()

        # --- Calculations ---
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

            # --- Matplotlib Pie Chart ---
            fig, ax = plt.subplots()
            ax.pie([win, loss], labels=[batting_team, bowling_team],
                   autopct='%1.1f%%', startangle=90,
                   colors=["#4CAF50", "#FF5722"], wedgeprops=dict(width=0.4))
            ax.set_title("📊 Win Probability", color="orange")
            st.pyplot(fig)

            # --- Results ---
            st.success(f"✅ {batting_team} has a `{round(win * 100)}%` chance to win!")
            st.info(f"❌ {bowling_team} has a `{round(loss * 100)}%` chance to win.")

            st.markdown("### 📈 Match Stats")
            st.markdown(f"**Runs Left:** `{runs_left}`")
            st.markdown(f"**Balls Left:** `{balls_left}`")
            st.markdown(f"**Wickets Remaining:** `{wickets_remaining}`")
            st.markdown(f"**Current Run Rate (CRR):** `{crr:.2f}`")
            st.markdown(f"**Required Run Rate (RRR):** `{rrr:.2f}`")

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
