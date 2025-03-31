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


#function for getting a players attribute nearest in time to the match_date
def get_player_attributes_closest_to_match_date(player_api_id, match_date): 
    player_rating_history = player_attr_df[player_attr_df["player_api_id"] == player_api_id]

    player_rating_history = player_rating_history.set_index("date").sort_index()

    iloc_idx = player_rating_history.index.get_indexer([match_date], method="nearest")
    loc_idx = player_rating_history.index[iloc_idx]
    closest_row = player_rating_history.loc[loc_idx]
    return closest_row.iloc[0]

match_df["home_team_rating_avrg"] = None
match_df["away_team_rating_avrg"] = None

#function for computing the home and away team_rating_avrgs of a row in the match_data_df
# Pretty slow process, takes roughly 6 min
def compute_team_ratings(row):
    match_date = row["date"]
    home_team_rating_avrg = 0
    away_team_rating_avrg = 0
    
    for i in range(11):
        home_player_api = row[f"home_player_{i+1}"]
        away_player_api = row[f"away_player_{i+1}"]

        home_player_attr_for_match = get_player_attributes_closest_to_match_date(home_player_api, match_date)
        away_player_attr_for_match = get_player_attributes_closest_to_match_date(away_player_api, match_date)

        home_team_rating_avrg += home_player_attr_for_match["overall_rating"]
        away_team_rating_avrg += away_player_attr_for_match["overall_rating"]
    
    row["home_team_rating_avrg"] = home_team_rating_avrg / 11
    row["away_team_rating_avrg"] = away_team_rating_avrg / 11
    return row


def get_prev_5_matches(row): #This can be reused for any features that involve calulcating both teams prev_5_avrg of something, 
    match_date = row["date"]
    home_team_api_id = row["home_team_api_id"]
    away_team_api_id = row["away_team_api_id"]

    home_team_matches = match_df[
        (match_df["away_team_api_id"] == home_team_api_id) | 
        (match_df["home_team_api_id"] == home_team_api_id)]
    
    away_team_matches = match_df[
        (match_df["away_team_api_id"] == away_team_api_id) | 
        (match_df["home_team_api_id"] == away_team_api_id)]

    home_past_matches = home_team_matches[home_team_matches["date"] < match_date]
    away_past_matches = away_team_matches[away_team_matches["date"] < match_date]

    home_past_matches = home_past_matches.sort_values(by="date", ascending=False)
    away_past_matches = away_past_matches.sort_values(by="date", ascending=False)

    return home_past_matches.head(5), away_past_matches.head(5)

match_df["home_prev_5_goal_diff"] = None
match_df["away_prev_5_goal_diff"] = None

def compute_prev_5_goal_diff(row):
    home_past_matches, away_past_matches = get_prev_5_matches(row)
    if home_past_matches.shape[0] != 5 or away_past_matches.shape[0] != 5:
        return row

    home_goal_diff = 0
    for _, match in home_past_matches.iterrows():
        if match["home_team_api_id"] == row["home_team_api_id"]:
            home_goal_diff += match["home_team_goal"] - match["away_team_goal"]
        else:
            home_goal_diff += match["away_team_goal"] - match["home_team_goal"]

    away_goal_diff = 0
    for _, match in away_past_matches.iterrows():
        if match["home_team_api_id"] == row["away_team_api_id"]:
            away_goal_diff += match["home_team_goal"] - match["away_team_goal"]
        else:
            away_goal_diff += match["away_team_goal"] - match["home_team_goal"]


    row["home_prev_5_goal_diff"] = home_goal_diff
    row["away_prev_5_goal_diff"] = away_goal_diff
    return row



match_df = match_df.apply(compute_prev_5_goal_diff, axis=1) #This will create some null values as if a team in the match has not played 5 games, it will not calculate the average

match_df = match_df.dropna(subset=["home_prev_5_goal_diff", "away_prev_5_goal_diff"]) 


# print(match_df)
    

#match_df = match_df.apply(compute_team_ratings, axis=1) #No point in running this unless we are creating the finalized set of features to be used


# print(match_df)