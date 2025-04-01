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
match_df = match_df.sort_values(by="date") #dunno if this is needed


match_df["date"] = pd.to_datetime(match_df["date"])
player_attr_df["date"] = pd.to_datetime(player_attr_df["date"])

home_player_cols = ["home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11"]

away_player_cols = ["away_player_1", "away_player_2", "away_player_3", "away_player_4","away_player_5",
                    "away_player_6", "away_player_7", "away_player_8", "away_player_9", "away_player_10",
                    "away_player_11" ]

match_df = match_df.drop(match_df.columns[11:55], axis=1) # DROP ALL COORDINATE COLUMNS

betting_odds_home = match_df.iloc[:, 41::3]
betting_oods_draw = match_df.iloc[:, 42::3]
betting_oods_away = match_df.iloc[:, 43::3]


match_df["betting_odds_home_avrg"] = betting_odds_home.mean(axis=1)
match_df["betting_odds_draw_avrg"] = betting_oods_draw.mean(axis=1)
match_df["betting_odds_away_avrg"] = betting_oods_away.mean(axis=1)


match_df = match_df.drop(match_df.columns[41:71], axis=1)

match_df = match_df.dropna(subset=["betting_odds_home_avrg",  "betting_odds_draw_avrg", "betting_odds_away_avrg"])

match_df = match_df.dropna(subset=home_player_cols)

match_df = match_df.dropna(subset=away_player_cols)

match_df = match_df.dropna(how="any", axis=0)

player_attr_df = player_attr_df.dropna(subset=["date", "overall_rating"])

team_games = {team: [] for team in team_df["team_api_id"]}
x = []
y = []

match_df["home_prev_5_goal_diff"] = None
match_df["away_prev_5_goal_diff"] = None

yes = 0

home_team_counts = match_df["home_team_api_id"].value_counts()
print(home_team_counts.shape[0])

for index, row in match_df.iterrows():
    home_players_stats = []
    away_players_stats = []
    match_date = row["date"]

    home_team = row["home_team_api_id"]
    away_team = row["away_team_api_id"]
    home_team_goal = row["home_team_goal"]
    away_team_goal = row["away_team_goal"]
    goal_diff_home = home_team_goal - away_team_goal
    goal_diff_away = away_team_goal - home_team_goal

    if (goal_diff_home  > 0):
        outcome = HOME_TEAM_WIN
    elif (goal_diff_home < 0):
        outcome = AWAY_TEAM_WIN
    else:
        outcome = DRAW

    home_last_5 = team_games[home_team][-5:]  
    away_last_5 = team_games[away_team][-5:]  

    if len(home_last_5) == 5 and len(away_last_5) == 5:
        features = []
        for i in range(5):
            h = home_last_5[i]
            a = away_last_5[i]
            features.append([h-a])
            match_df.loc[index, "home_prev_5_goal_diff"] = 1
            match_df.loc[index, "away_prev_5_goal_diff"] = 1
        
        yes +=1
        x.append(features)  
        y.append(outcome)  


    team_games[home_team].append(goal_diff_home)
    team_games[away_team].append(goal_diff_away) 


for key in team_games:
    print(len(team_games[key]))

print(len(team_games))
print(yes)

print(match_df["home_prev_5_goal_diff"].isnull().sum())

# def test(row):
#     home_team = row["home_team_api_id"]
#     away_team = row["away_team_api_id"]


# match_df = match_df.apply(test, axis=1)    

# print(x)
# print(team_games)

# x = np.array(x)
# y = np.array(y)


# x = x.reshape(x.shape[0], 5, 1)
# print(x.shape)


##DATA NEEDS TO BE SCALED WITH MinMaxScaler

# y_categorical = keras.utils.to_categorical(y, num_classes=3)

# model = keras.Sequential([
#     keras.layers.LSTM(50, return_sequences=False),
#     keras.layers.Dense(20, activation="relu"),
#     keras.layers.Dense(3, activation="softmax")  
# ])


# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# model.fit(x, y_categorical, epochs=10, batch_size=32, validation_split=0.2)