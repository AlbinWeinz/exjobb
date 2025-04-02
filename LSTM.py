import numpy as np
import pandas as pd
import sqlite3 as db
from pathlib import Path
import keras
from sklearn.preprocessing import MinMaxScaler

NUM_CLASSES = 3
HOME_TEAM_WIN = 1
AWAY_TEAM_WIN = 2
DRAW = 0

current_dir = Path(__file__).parent
input_path = current_dir / "input" / "features.csv"

input = pd.read_csv(input_path)

teams = unique_values = set(input["home_team_api_id"]).union(set(input["away_team_api_id"]))


team_games = {team: [] for team in teams}
x = []
y = []

# x_cols = ["home_team_api_id", "away_team_api_id", "betting_odds_home_avrg", "betting_odds_draw_avrg" ,
#                 "betting_odds_away_avrg", "home_team_rating_avrg", "away_team_rating_avrg", "home_prev_5_goal_diff", 
#                 "away_prev_5_goal_diff", "on_target_shot_home_team", "on_target_shot_away_team", "off_target_shot_home_team", 
#                 "off_target_shot_away_team", "foul_home_team", "foul_away_team", "yellow_card_home_team",
#                 "yellow_card_away_team", "red_card_home_team", "red_card_away_team", "crosses_home_team", "crosses_away_team",
#                 "corner_home_team", "corner_away_team", "possession_home_team", "possession_away_team"] 


x_cols = ["betting_odds_home_avrg", "betting_odds_draw_avrg" ,
                "betting_odds_away_avrg", "home_team_rating_avrg", "away_team_rating_avrg", "home_prev_5_goal_diff", 
                "away_prev_5_goal_diff", "on_target_shot_home_team", "on_target_shot_away_team", "off_target_shot_home_team", 
                "off_target_shot_away_team", "foul_home_team", "foul_away_team", "yellow_card_home_team",
                "yellow_card_away_team", "red_card_home_team", "red_card_away_team", "crosses_home_team", "crosses_away_team",
                "corner_home_team", "corner_away_team", "possession_home_team", "possession_away_team"] 

home_cols = [col for col in x_cols if "home" in col]
away_cols = [col for col in x_cols if "away" in col]


x = input[x_cols].values.tolist()
y = input["match_outcome"].values.tolist()
# print(x)
# for _, row in input.iterrows():

#     home_team = row["home_team_api_id"]
#     away_team = row["away_team_api_id"]

#     home_last_5 = team_games[home_team][-5:]  
#     away_last_5 = team_games[away_team][-5:]  

#     if len(home_last_5) == 5 and len(away_last_5) == 5:
#         features = []
#         for i in range(5):
#             # h = home_last_5[i]
#             # a = away_last_5[i]
#             # features.append(h + a)
            
#         x.append(features)  
#         y.append(row["match_outcome"])  

#     x.appen()


#     team_games[home_team].append(row[home_cols].tolist())
#     team_games[away_team].append(row[away_cols].tolist()) 

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], 1, 23)


y_categorical = keras.utils.to_categorical(y, num_classes=3)

model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(3, activation="softmax")  
])

adm = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss="categorical_crossentropy", optimizer=adm, metrics=["accuracy"])

model.fit(x, y_categorical, epochs=10, batch_size=64, validation_split=0.2)