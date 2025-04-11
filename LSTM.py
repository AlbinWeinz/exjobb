import numpy as np
import pandas as pd
from pathlib import Path
import keras
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
from sklearn.metrics import confusion_matrix


current_dir = Path(__file__).parent
input_path = current_dir / "input" / "features.csv"

input = pd.read_csv(input_path)
teams = set(input["home_team_api_id"]).union(set(input["away_team_api_id"]))
team_games = {team: [] for team in teams}
x = []
y = []



feature_cols = ["betting_odds_home_avrg", "betting_odds_draw_avrg" ,
                "betting_odds_away_avrg", "home_team_rating_avrg", "away_team_rating_avrg", "home_prev_5_goal_diff", 
                "away_prev_5_goal_diff", "on_target_shot_home_team", "on_target_shot_away_team", "off_target_shot_home_team", 
                "off_target_shot_away_team", "foul_home_team", "foul_away_team", "yellow_card_home_team",
                "yellow_card_away_team", "red_card_home_team", "red_card_away_team", "crosses_home_team", "crosses_away_team",
                "corner_home_team", "corner_away_team", "possession_home_team", "possession_away_team"] 

home_feature_cols = [col for col in feature_cols if "home" in col or "odds" in col] 
away_feature_cols = [col for col in feature_cols if "away" in col or "odds" in col]
scaler = MinMaxScaler(feature_range=(0, 1)) 
input[feature_cols] = scaler.fit_transform(input[feature_cols]) 


for _, row in input.iterrows():

    home_team = row["home_team_api_id"]
    away_team = row["away_team_api_id"]

    home_last_5 = team_games[home_team][-5:]  
    away_last_5 = team_games[away_team][-5:]  

    if len(home_last_5) == 5 and len(away_last_5) == 5:
        window = []
        for i in range(5):
            home_features = home_last_5[i]
            away_features = away_last_5[i]
            window.append(home_features + away_features)
            
        x.append(window)  
        y.append(row["match_outcome"])  

    team_games[home_team].append(row[home_feature_cols].tolist())
    team_games[away_team].append(row[away_feature_cols].tolist()) 

x = np.array(x)
y = np.array(y)
num_samples = x.shape[0]
train_end = int(num_samples * 0.7)
val_end = int(num_samples * 0.85)
x_train = x[:train_end]
y_train = y[:train_end]
x_val = x[train_end:val_end]
y_val = y[train_end:val_end]
x_test = x[val_end:]
y_test = y[val_end:]

# num_draw = 0
# num_home = 0
# num_away = 0
# for c in range(len(y_test)):
#     if y_test[c] == 0:
#         num_draw+=1
#     elif y_test[c] == 1:
#         num_home+=1
#     else:
#         num_away+=1

# print(num_draw, num_home, num_away)

x_train = x_train.reshape(x_train.shape[0], 5, 26)
x_val = x_val.reshape(x_val.shape[0], 5, 26)
x_test = x_test.reshape(x_test.shape[0], 5, 26)
y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_val = keras.utils.to_categorical(y_val, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))
    hp_learning_rate = hp.Float(
            "learning_rate", min_value=1e-4,
            max_value=1, sampling="log")
    rmsprop = keras.optimizers.RMSprop(
        learning_rate=hp_learning_rate)
    model.compile(optimizer=rmsprop,
                    loss="categorical_crossentropy", 
                    metrics=["accuracy", "precision",
                              "recall", "f1_score"])
    return model
tuner = kt.RandomSearch(build_model, objective="val_loss",
                         overwrite=True, max_trials=5)
tuner.search(x_train, y_train,
              validation_data=(x_val, y_val), epochs=10)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hp)
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=5,
    restore_best_weights=True
)
model.fit(x_train, y_train, validation_data=(x_val, y_val),
           epochs=50, callbacks=early_stopping, shuffle=False)

test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)
print("Best LR", best_hp["learning_rate"])
print("Test accuracy:", test_accuracy)
print("Test precision:", test_precision)
print("Test recall:", test_recall)
print("Test F1-score:", test_f1)

y_prediction = model.predict(x_test)
y_prediction = np.argmax (y_prediction, axis = 1)
y_test=np.argmax(y_test, axis=1)
c_matrix = confusion_matrix(y_test, y_prediction , normalize="pred")
print(c_matrix)

