import numpy as np
import pandas as pd
import sqlite3 as db
from pathlib import Path
import datetime


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


# print(match_df["shoton"].isnull().sum())
# print(match_df["shotoff"].isnull().sum())
# print(match_df["foulcommit"].isnull().sum())
# print(match_df["cross"].isnull().sum())
# print(match_df["corner"].isnull().sum())
# print(match_df["possession"].isnull().sum())
# print(match_df[["card", "shoton"]].isnull().sum())
# print(match_df.shape[0])
# match_df = match_df.dropna(how="any", axis=0)

# print(match_df.shape[0])

# print(match_df.isnull().sum())

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

print(match_df)

match_df = match_df.drop(match_df.columns[41:71], axis=1)

match_df = match_df.dropna(subset=["betting_odds_home_avrg",  "betting_odds_draw_avrg", "betting_odds_away_avrg"])

match_df = match_df.dropna(subset=home_player_cols)

match_df = match_df.dropna(subset=away_player_cols)

match_df = match_df.dropna(how="any", axis=0)

print(match_df.isna().sum())
print(match_df.shape[0]) #This creates a dataset with 13221 matches, with 0 missing values

# print(match_df["betting_avg_home"].notnull().sum())

# print(match_df["betting_avg_home"].isnull().sum())

# betting_df = match_df[betting_cols]

# print(betting_df.shape[0])
# # betting_df = betting_df.dropna(how="all", axis=0)
# print(betting_df.shape[0])
# print(betting_df)
# print(betting_df)
# match_df = match_df[[
#     "home_player_1","home_player_2","home_player_3","home_player_4","home_player_5","home_player_6","home_player_7","home_player_8","home_player_9","home_player_10","home_player_11",
#     "away_player_1", "away_player_2","away_player_3","away_player_4","away_player_5","away_player_6","away_player_7","away_player_8","away_player_9","away_player_10","away_player_11"]].dropna(how="any", axis=0)

# print(match_df[[
#     "home_player_1","home_player_2","home_player_3","home_player_4","home_player_5","home_player_6","home_player_7","home_player_8","home_player_9","home_player_10","home_player_11",
#     "away_player_1", "away_player_2","away_player_3","away_player_4","away_player_5","away_player_6","away_player_7","away_player_8","away_player_9","away_player_10","away_player_11"]].isnull().sum())
# print(match_df.shape)

# print(match_df.isnull().sum())

# team_games = {team: [] for team in match_df["home_team_api_id"].unique()}

# team_games2 = {team: [] for team in team_df["team_api_id"]}

# print(len(team_games))
# print(len(team_games2))



# print(player_attr_df.isnull().sum())
# print(player_attr_df.size)
# player_attr_df = player_attr_df.dropna(how="any", axis=0)
# print(player_attr_df.isnull().sum())
# print(player_attr_df)

# print(match_df["home_player_1"].isnull().sum())


# history.
# print(history["date"])

# iloc_idx = history.index.get_indexer([x], method="nearest")
# val = history[iloc_idx]
# print(val)
# print(player_history)