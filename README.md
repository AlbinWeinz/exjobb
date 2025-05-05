This project was undertaken and completed as a thesis for the DVK Uppsats VT2025 program at Stockholm University. Its primary objective was to compare the predictive performance of a Random Forest model, optimized for its minimum number of samples per leaf, with a Long Short-Term Memory (LSTM) model, optimized for its learning rate, for the task of classifying football matches as a home win, draw, or away win. An additional objective was to analyze the effect of each model’s key hyperparameter on performance.

To complete the data preparation phase, contained in data_preparation.py, the program requires the file "database.sqlite" to be located in the input folder. This file can be downloaded from:
https://www.kaggle.com/datasets/hugomathien/soccer/data.

To test the models, either LSTM.py or random_forest.py may be executed. This step does not require running data_preparation.py, as the processed model input is already available in the "features.csv" file
