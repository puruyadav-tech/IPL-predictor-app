import streamlit as st
import pickle
import pandas as pd

# Load model
pipe = pickle.load(open('mdl.pkl', 'rb'))

teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

logo_urls = {
    "Chennai Super Kings": "https://upload.wikimedia.org/wikipedia/en/2/2d/Chennai_Super_Kings_Logo.png",
    "Delhi Capitals": "https://upload.wikimedia.org/wikipedia/en/d/dc/Delhi_Capitals.svg",
    "Kings XI Punjab": "https://upload.wikimedia.org/wikipedia/en/d/d4/Kings_XI_Punjab_Logo.svg",
    "Kolkata Knight Riders": "https://upload.wikimedia.org/wikipedia/en/4/4f/Kolkata_Knight_Riders_Logo.svg",
    "Mumbai Indians": "https://upload.wikimedia.org/wikipedia/en/2/25/Mumbai_Indians_Logo.svg",
    "Rajasthan Royals": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg",
    "Royal Challengers Bangalore": "https://upload.wikimedia.org/wikipedia/en/4/4c/Royal_Challengers_Bangalore_Logo.svg",
    "Sunrisers Hyderabad": "https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad_Logo.svg"
}

st.title('🏏 IPL Win Predictor')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

if batting_team in logo_urls:
    st.image(logo_urls[batting_team], width=100, caption="Batting Team")
if bowling_team in logo_urls:
    st.image(logo_urls[bowling_team], width=100, caption="Bowling Team")

selected_city = st.selectbox('Select Host City', sorted(cities))
target = st.number_input('Target Score', step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', step=1)
with col4:
    overs = st.number_input('Overs Completed', step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', step=1)

if st.button('Predict Probability'):
    try:
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

        st.subheader(f"✅ {batting_team} - {round(win * 100)}% chance to win")
        st.subheader(f"❌ {bowling_team} - {round(loss * 100)}% chance to win")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
