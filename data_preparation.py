import pandas as pd
import sqlite3 as db
from pathlib import Path
import xml.etree.ElementTree as ET

HOME_TEAM_WIN = 1
AWAY_TEAM_WIN = 2
DRAW = 0

current_dir = Path(__file__).parent
input_path = current_dir / "input" / "database.sqlite"

conn = db.connect(input_path)

match_df = pd.read_sql_query("SELECT * FROM Match", conn)
player_attr_df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
match_df = match_df.sort_values(by="date")

match_df["date"] = pd.to_datetime(match_df["date"])
player_attr_df["date"] = pd.to_datetime(player_attr_df["date"])

match_df = match_df.drop(match_df.columns[11:55], axis=1) 
match_df = match_df.drop(["stage", "match_api_id", "country_id", "league_id", "season", "id"], axis=1) 

# Function to extract and separate XML stats for home team and away team
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
match_df = match_df.drop(match_df.columns[27:35], axis=1)


#Betting features
betting_odds_home = match_df.iloc[:, 27:55:3] 
betting_odds_draw = match_df.iloc[:, 28:56:3] 
betting_odds_away = match_df.iloc[:, 29:57:3]
match_df["betting_odds_home_avrg"] = betting_odds_home.mean(axis=1)
match_df["betting_odds_draw_avrg"] = betting_odds_draw.mean(axis=1)
match_df["betting_odds_away_avrg"] = betting_odds_away.mean(axis=1)
match_df = match_df.drop(match_df.columns[27:57], axis=1)
match_df = match_df.dropna(subset=["betting_odds_home_avrg",  "betting_odds_draw_avrg", "betting_odds_away_avrg"])

player_attr_df = player_attr_df.dropna(subset=["date", "overall_rating"])
#function for getting a players attribute nearest in time to the match_date
def get_player_attr_nearest_date(player_api_id, match_date): 
    player_rating_history = player_attr_df[player_attr_df["player_api_id"] == player_api_id]
    player_rating_history = player_rating_history.set_index("date").sort_index()
    iloc_idx = player_rating_history.index.get_indexer([match_date], method="nearest")
    loc_idx = player_rating_history.index[iloc_idx]
    closest_row = player_rating_history.loc[loc_idx]
    return closest_row.iloc[0]

# Team rating feature
player_columns = match_df.columns[5:27]
match_df = match_df.dropna(how="any", subset=player_columns)
match_df["home_team_rating_avrg"] = None
match_df["away_team_rating_avrg"] = None
#Function for computing the home and away team ratings
def compute_team_ratings(row):
    match_date = row["date"]
    home_team_rating_avrg = 0
    away_team_rating_avrg = 0   
    for i in range(11):
        home_player_api = row[f"home_player_{i+1}"]
        away_player_api = row[f"away_player_{i+1}"]
        home_player_attr = get_player_attr_nearest_date(
            home_player_api, match_date)
        away_player_attr = get_player_attr_nearest_date(
            away_player_api, match_date)
        home_team_rating_avrg += home_player_attr["overall_rating"]
        away_team_rating_avrg += away_player_attr["overall_rating"]
    row["home_team_rating_avrg"] = home_team_rating_avrg / 11
    row["away_team_rating_avrg"] = away_team_rating_avrg / 11
    return row
match_df = match_df.apply(compute_team_ratings, axis=1) 
match_df = match_df.drop(match_df.columns[5:27], axis=1)

#Function for getting home and away teams previous 5 games
def get_prev_5_matches(row): 
    match_date = row["date"]
    home_team_api_id = row["home_team_api_id"]
    away_team_api_id = row["away_team_api_id"]

    home_team_matches = match_df[
        (match_df["away_team_api_id"] == home_team_api_id) | 
        (match_df["home_team_api_id"] == home_team_api_id)]
    
    away_team_matches = match_df[
        (match_df["away_team_api_id"] == away_team_api_id) | 
        (match_df["home_team_api_id"] == away_team_api_id)]

    home_past_5 = home_team_matches[home_team_matches["date"] < match_date]
    away_past_5 = away_team_matches[away_team_matches["date"] < match_date]

    home_past_5 = home_past_5.sort_values(by="date", ascending=False)
    away_past_5 = away_past_5.sort_values(by="date", ascending=False)

    return home_past_5.head(5), away_past_5.head(5)

match_df["home_prev_5_goal_diff"] = None
match_df["away_prev_5_goal_diff"] = None
#Function for computing prev 5 games goal difference for home and away team
def compute_prev_5_goal_diff(row):
    home_past_5, away_past_5 = get_prev_5_matches(row)
    if home_past_5.shape[0] != 5 or away_past_5.shape[0] != 5:
        return row

    home_goal_diff = 0
    for _, match in home_past_5.iterrows():
        if match["home_team_api_id"] == row["home_team_api_id"]:
            home_goal_diff += (match["home_team_goal"]
                                - match["away_team_goal"])
        else:
            home_goal_diff += (match["away_team_goal"] 
                               - match["home_team_goal"])

    away_goal_diff = 0
    for _, match in away_past_5.iterrows():
        if match["home_team_api_id"] == row["away_team_api_id"]:
            away_goal_diff += match["home_team_goal"] - match["away_team_goal"]
        else:
            away_goal_diff += match["away_team_goal"] - match["home_team_goal"]


    row["home_prev_5_goal_diff"] = home_goal_diff
    row["away_prev_5_goal_diff"] = away_goal_diff
    return row
match_df = match_df.apply(compute_prev_5_goal_diff, axis=1)

#Function for computing previous 5 mean stats
def compute_prev_5_mean_stats(row):
    cols = match_df.columns[5:21]
    for home_col in cols[::2]:
        home_col_index = match_df.columns.get_loc(home_col)
        away_col = match_df.columns[home_col_index + 1]
        home_past_5, away_past_5 = get_prev_5_matches(row)
        if home_past_5.shape[0] != 5 or away_past_5.shape[0] != 5:
            row[home_col] = None
            row[away_col] = None
        else:
            home_avrg = 0
            for _, match in home_past_5.iterrows():
                if (match["home_team_api_id"]
                        == row["home_team_api_id"]):
                    home_avrg += match[home_col]
                else:
                    home_avrg += match[away_col]
            home_avrg = home_avrg / 5

            away_avrg = 0
            for _, match in away_past_5.iterrows():
                if match[
                    "home_team_api_id"] == row["away_team_api_id"]:
                    away_avrg += match[home_col]
                else:
                    away_avrg += match[away_col]
            away_avrg = away_avrg / 5

            row[home_col] = home_avrg
            row[away_col] = away_avrg
    return row
match_df = match_df.apply(compute_prev_5_mean_stats, axis=1) 
match_df = match_df.dropna(how="any", axis=0)

match_df["match_outcome"] = None
#Function for computing match outcome
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
match_df = match_df.apply(compute_match_outcome, axis=1) 
match_df = match_df.drop(labels=["home_team_goal", "away_team_goal"], axis=1)

output_path = current_dir / "input" / "features.csv"
match_df.to_csv(output_path, index=False)