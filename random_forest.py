import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

match_df = pd.read_csv("input/features.csv")

features = [
    'betting_odds_home_avrg', 'betting_odds_draw_avrg', 'betting_odds_away_avrg',
    'home_team_rating_avrg', 'away_team_rating_avrg', 'home_prev_5_goal_diff', 'away_prev_5_goal_diff',
    'on_target_shot_home_team', 'on_target_shot_away_team', 'off_target_shot_home_team', 'off_target_shot_away_team', 'foul_home_team',
    'foul_away_team', 'yellow_card_home_team', 'yellow_card_away_team', 'red_card_home_team', 'red_card_away_team', 'crosses_home_team',
    'crosses_away_team', 'corner_home_team', 'corner_away_team', 'possession_home_team', 'possession_away_team'
]

X = match_df[features]
y = match_df['match_outcome']

train_partition = 0.7
val_partition = 0.15
test_partition = 0.15

train_split = int(len(match_df) * train_partition)
val_split = train_split + int(len(match_df) * val_partition)

train_df = match_df.iloc[:train_split]
val_df = match_df.iloc[train_split:val_split]
test_df = match_df.iloc[val_split:]

X_train, y_train = train_df[features], train_df["match_outcome"]
X_val, y_val = val_df[features], val_df["match_outcome"]
X_test, y_test = test_df[features], test_df["match_outcome"]

base_model = RandomForestClassifier(random_state=42)

# Set up random search for min_samples_leaf
param_grid = {
    'min_samples_leaf': list(range(1, 51))  # 1 to 50
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid
)

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

y_test_predict = model.predict(X_test)

report = classification_report(y_test, y_test_predict, digits=4)

print(report)

results = grid_search.cv_results_
min_samples = results['param_min_samples_leaf'].data
mean_scores = results['mean_test_score']


plt.figure(figsize=(10, 6))
plt.plot(min_samples, mean_scores, marker='o')
plt.xlabel('Min_Samples_Leaf')
plt.ylabel('Accuracy')
plt.title('Min_Samples_Leaf effect on model Accuracy')
plt.grid(True)
plt.show()





