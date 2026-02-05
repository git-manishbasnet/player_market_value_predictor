# app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ========================
# 1️⃣ Load Dataset & Train Model
# ========================
df = pd.read_csv("player_value_50.csv")  # your 50-player CSV

# Features and target
X = df[["Age","Matches","Goals","Assists"]]
y = df["MarketValue_Million_EUR"]  # 1D target
model = LinearRegression()
model.fit(X, y)

# ========================
# 2️⃣ Streamlit UI
# ========================
st.title("Football Player Market Value Predictor ⚽")

# Player selection from dataset
player_name = st.selectbox("Select Player", df["PlayerName"].tolist())

# Get player stats from dataset
player_stats = df[df["PlayerName"] == player_name][["Age","Matches","Goals","Assists"]]

# Predict button
if st.button("Predict Market Value"):
    prediction = model.predict(player_stats)
    st.success(f"Predicted Market Value for {player_name}: € {prediction[0]:.2f} Million")

# ========================
# 3️⃣ Optional: Custom Input
# ========================
st.subheader("Or Predict for a Custom Player")

age = st.number_input("Age", min_value=16, max_value=40, value=24)
matches = st.number_input("Matches Played", min_value=0, max_value=100, value=30)
goals = st.number_input("Goals Scored", min_value=0, max_value=50, value=15)
assists = st.number_input("Assists", min_value=0, max_value=50, value=10)

if st.button("Predict Custom Player"):
    custom_stats = [[age, matches, goals, assists]]
    custom_prediction = model.predict(custom_stats)
    st.success(f"Predicted Market Value for Custom Player: € {custom_prediction[0]:.2f} Million")
