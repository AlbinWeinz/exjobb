import numpy as np
import pandas as pd
import sqlite3 as db
from pathlib import Path
import datetime
import xml.etree.ElementTree as ET


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

match_df["match_outcome"] = None

def compute_match_outcome(row):
    home_team_goal = row["home_team_goal"]
    away_team_goal = row["away_team_goal"]
    if home_team_goal > away_team_goal:
        row["match_outcome"] = HOME_TEAM_WIN
    elif home_team_goal < away_team_goal:
        row["match_outcome"] = AWAY_TEAM_WIN
    else:
        row["match_outcome"] = DRAW

    return row



match_df = match_df.apply(compute_prev_5_goal_diff, axis=1) #This will create some null values as if a team in the match has not played 5 games, it will not extract the average


# Function to separate stats for home team and away team
def extract_stats_both_teams(xml_document, home_team, away_team, card_type='y'):
    if not xml_document or xml_document.strip() == "":
        return 'None', 'None'  # Default values

    try:
        tree = ET.fromstring(xml_document)  # Parse XML
    except ET.ParseError:
        return 'None', 'None'  # If XML is invalid, return 0

    stat_home_team = 0
    stat_away_team = 0

    if tree.tag == 'card':
        for child in tree.iter('value'):
            # Some xml docs have no card_type element in the tree. comment section seems to have that information
            try:
                if child.find('comment').text == card_type:
                    if int(child.find('team').text) == home_team:
                        stat_home_team += 1
                    else:
                        stat_away_team += 1
            except AttributeError:
                # Some values in the xml doc don't have team values, so there isn't much we can do at this stage
                pass

        return stat_home_team, stat_away_team

        # Lets take the last possession stat which is available from the xml doc
    if tree.tag == 'possession':
        try:
            last_value = [child for child in tree.iter('value')][-1]
            return int(last_value.find('homepos').text), int(last_value.find('awaypos').text)
        except:
            return None, None

    for team in [int(stat.text) for stat in tree.findall('value/team')]:
        if team == home_team:
            stat_home_team += 1
        else:
            stat_away_team += 1

    return stat_home_team, stat_away_team

# Adds columns for stats separated for home team and away team
match_df[['on_target_shot_home_team','on_target_shot_away_team']] = match_df[['shoton','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['shoton'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
match_df[['off_target_shot_home_team','off_target_shot_away_team']] = match_df[['shotoff','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['shotoff'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
match_df[['foul_home_team','foul_away_team']] = match_df[['foulcommit','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['foulcommit'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
match_df[['yellow_card_home_team','yellow_card_away_team']] = match_df[['card','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['card'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
match_df[['red_card_home_team','red_card_away_team']] = match_df[['card','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['card'],x['home_team_api_id'],x['away_team_api_id'], card_type='r'), axis = 1,result_type="expand")
match_df[['crosses_home_team','crosses_away_team']] = match_df[['cross','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['cross'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
match_df[['corner_home_team','corner_away_team']] = match_df[['corner','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['corner'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")
match_df[['possession_home_team','possession_away_team']] = match_df[['possession','home_team_api_id','away_team_api_id']].apply(lambda x: extract_stats_both_teams(x['possession'],x['home_team_api_id'],x['away_team_api_id']), axis = 1,result_type="expand")

# Filters out all null values
match_df = match_df[
    (match_df["on_target_shot_home_team"] != 'None') &
    (match_df["off_target_shot_home_team"] != 'None') &
    (match_df["foul_home_team"] != 'None') &
    (match_df["yellow_card_home_team"] != 'None') &
    (match_df["red_card_home_team"] != 'None') &
    (match_df["crosses_home_team"] != 'None') &
    (match_df["corner_home_team"] != 'None') &
    (match_df["possession_home_team"] != 'None') &
    (match_df["possession_home_team"].notna())
]


def compute_prev_5_avrges(row):
    cols = match_df.columns[49:]
    for home_col in cols[::2]:
        home_col_index = match_df.columns.get_loc(home_col)
        away_col = match_df.columns[home_col_index + 1]
        home_past_matches, away_past_matches = get_prev_5_matches(row)
        if home_past_matches.shape[0] != 5 or away_past_matches.shape[0] != 5:
            row[home_col] = None
            row[away_col] = None
            return row

        home_avrg = 0
        for _, match in home_past_matches.iterrows():
            if match["home_team_api_id"] == row["home_team_api_id"]:
                home_avrg += match[home_col]
            else:
                home_avrg += match[away_col]
        home_avrg = home_avrg / 5

        away_avrg = 0
        for _, match in away_past_matches.iterrows():
            if match["home_team_api_id"] == row["away_team_api_id"]:
                away_avrg += match[home_col]
            else:
                away_avrg += match[away_col]
        away_avrg = away_avrg / 5

        row[home_col] = home_avrg
        row[away_col] = away_avrg
    return row


match_df = match_df.apply(compute_match_outcome, axis=1)

match_df = match_df.apply(compute_prev_5_avrges, axis=1) 

match_df = match_df.dropna(subset=["home_prev_5_goal_diff", "away_prev_5_goal_diff"]) 



match_df = match_df.apply(compute_team_ratings, axis=1) #No point in running this unless we are creating the finalized set of features to be used


print(match_df)


match_df = match_df.drop(match_df.columns[9:41], axis=1)
match_df = match_df.drop(["stage", "match_api_id"], axis=1)
output_path = current_dir / "input" / "features.csv"
match_df.to_csv(output_path, index=False)

print(match_df)