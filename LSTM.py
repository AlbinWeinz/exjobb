import numpy as np
import pandas as pd
import sqlite3 as db
from pathlib import Path
import keras

NUM_CLASSES = 3
HOME_TEAM_WIN = 1
AWAY_TEAM_WIN = 2
DRAW = 0

current_dir = Path(__file__).parent
input_path = current_dir / "input" / "database.sqlite"

conn = db.connect(input_path)

country_df = pd.read_sql_query("SELECT * FROM Country", conn)
league_df = pd.read_sql_query("SELECT * FROM League", conn)
team_df = pd.read_sql_query("SELECT * FROM Team", conn)
match_df = pd.read_sql_query("SELECT * FROM Match", conn)
player_df = pd.read_sql_query("SELECT * FROM Player", conn)
player_attr_df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
team_attr_df = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
match_df = match_df.sort_values(by="date")

# match_df = match_df.dropna(how="any", axis=0) # DROPS ALL ROWS WITH MISSING VALUES, THIS REDUCES SIZE FROM 25979 ROWS TO 1762, need better cleaning
player_attr_df = player_attr_df.dropna(how="any", axis=0)
match_df["date"] = pd.to_datetime(match_df["date"])
player_attr_df["date"] = pd.to_datetime(player_attr_df["date"])

team_games = {team: [] for team in team_df["team_api_id"]}
x = []
y = []

for _, row in match_df.iterrows():
    home_players_stats = []
    away_players_stats = []
    match_date = row["date"]

    home_team = row["home_team_api_id"]
    away_team = row["away_team_api_id"]
    home_team_goal = row["home_team_goal"]
    away_team_goal = row["away_team_goal"]
    goal_diff = home_team_goal - away_team_goal


    if (goal_diff  > 0):
        outcome = HOME_TEAM_WIN
    elif (goal_diff < 0):
        outcome = AWAY_TEAM_WIN
    else:
        outcome = DRAW



    home_last_5 = team_games[home_team][-5:]  
    away_last_5 = team_games[away_team][-5:]  


    if len(home_last_5) == 5 and len(away_last_5) == 5:
        features = []
        for i in range(5):
            h = home_last_5[i]
            # print(h)
            
            # print(h.get("goal_diff"))
            a = away_last_5[i]



            # features.append([h[0]-a[0], h[1]-a[1]])
            features.append([h-a])


        
        x.append(features)  
        y.append(outcome)  


    # team_games[home_team].append([goal_diff, rating_diff])
    # team_games[away_team].append([goal_diff, rating_diff]) 
    team_games[home_team].append(goal_diff)
    team_games[away_team].append(goal_diff) 


# print(x)
# print(team_games)

x = np.array(x)
y = np.array(y)


x = x.reshape(x.shape[0], 5, 1)
print(x.shape)

y_categorical = keras.utils.to_categorical(y, num_classes=3)

model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(3, activation="softmax")  
])


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(x, y_categorical, epochs=10, batch_size=32, validation_split=0.2)