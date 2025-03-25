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

# country_df = pd.read_sql_query("SELECT * FROM Country", conn)
# league_df = pd.read_sql_query("SELECT * FROM League", conn)
# team_df = pd.read_sql_query("SELECT * FROM Team", conn)
match_df = pd.read_sql_query("SELECT * FROM Match", conn)
# player_df = pd.read_sql_query("SELECT * FROM Player", conn)
# player_attr_df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
# team_attr_df = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
sorted_by_date = match_df.sort_values(by="date")

team_games = {team: [] for team in sorted_by_date["home_team_api_id"].unique()}
x = []
y = []

for _, row in sorted_by_date.iterrows():
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
        goal_diff = []
        for goal in range(5):
            h = home_last_5[goal]
            a = away_last_5[goal]
            goal_diff.append([h-a])

        
        x.append(goal_diff)  
        y.append(outcome)  


    team_games[home_team].append(home_team_goal)
    team_games[away_team].append(away_team_goal)  




x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], 5, 1)

y_categorical = keras.utils.to_categorical(y, num_classes=3)

model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=False, input_shape=(10, 1)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(3, activation="softmax")  
])


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(x, y_categorical, epochs=10, batch_size=32, validation_split=0.2)