# %%
import nba_api.stats
import nba_api.stats.static
import nba_api.stats.static.teams
import pandas as pd
import numpy as np
import sqlite3
import nba_api

pd.set_option('display.max_columns', 500)
# Connect to the SQLite database
conn = sqlite3.connect('nba.sqlite')

# Query the game table and load the data into a DataFrame
query = """
SELECT *
FROM game
"""
games = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()
games

# %%
team_id_age_arr = np.array([[str(t.get('id')), t.get('year_founded')] for t in nba_api.stats.static.teams.get_teams()])

# Prepare the data
df = games.copy()
msk = (df['team_id_home'].isin(team_id_age_arr[:, 0])) | (df['team_id_away'].isin(team_id_age_arr[:, 0]))

df = df.drop_duplicates(subset=['game_id'])
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.loc[df['season_type'].isin(["Playoffs", "Regular Season"])]
df = df.loc[df['game_date']>="1990-08-01"]
# Sorting to ensure correct rolling calculation
df = df.sort_values(by='game_date')

# Calculate days since last game for home and away teams
df['days_since_last_game_home'] = df.groupby('team_id_home')['game_date'].diff().dt.days
df['days_since_last_game_away'] = df.groupby('team_id_away')['game_date'].diff().dt.days

# Calculate team age at time of game
team_age_df = pd.DataFrame(team_id_age_arr, columns=['team_id', 'year_founded'])
team_age_df['year_founded'] = team_age_df['year_founded'].astype(int)

df = df.merge(team_age_df, left_on='team_id_home', right_on='team_id', how='left')
df['team_age_at_game_home'] = df['game_date'].dt.year - df['year_founded']
df = df.drop(columns=['team_id', 'year_founded'])

df = df.merge(team_age_df, left_on='team_id_away', right_on='team_id', how='left')
df['team_age_at_game_away'] = df['game_date'].dt.year - df['year_founded']
df = df.drop(columns=['team_id', 'year_founded'])

# Rolling statistics for home team
rolling_features = ['fgm_home', 'fga_home', 'fg_pct_home', 'fg3m_home', 'fg3a_home', 'fg3_pct_home', 'ftm_home', 'fta_home', 'ft_pct_home', 'oreb_home', 'dreb_home', 'reb_home', 'ast_home', 'stl_home', 'blk_home', 'tov_home', 'pf_home', 'pts_home']
for feature in rolling_features:
    df[f'rolling_avg_3_{feature}'] = df.groupby('team_id_home')[feature].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Rolling statistics for away team
for feature in rolling_features:
    modified_feature = feature.replace('home', 'away')
    df[f'rolling_avg_3_{modified_feature}'] = df.groupby('team_id_away')[modified_feature].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Calculate the differences between rolling averages of home and away teams for comparative analysis
for feature in rolling_features:
    home_feature = f'rolling_avg_3_{feature}'
    away_feature = home_feature.replace('home', 'away')
    df[f'diff_{feature}'] = df[home_feature] - df[away_feature]

# Define the outcome variables
df['home_win'] = (df['wl_home'] == 'W').astype(int)  # 1 if home team won, 0 otherwise
df['point_diff'] = df['pts_home'] - df['pts_away']  # Point difference of the game

# Select only the necessary columns for training
model_features = [col for col in df.columns if 'rolling_avg_3' in col or col in ['home_win', 'point_diff', 'game_date', 'season_id', 'days_since_last_game_home', 'days_since_last_game_away', 'team_age_at_game_home', 'team_age_at_game_away']]
final_df = df[model_features]

home_features = [c for c in final_df.columns if "home" in c and not c=="home_win"]
away_features = [c for c in final_df.columns if "away" in c and not c=="home_win"]
final_df

# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Normalization
from tensorflow.keras.utils import plot_model

tf.keras.backend.clear_session()

train_df = final_df.copy()

# Assuming df is your DataFrame loaded from the SQLite database
# Fill NaN values with the mean of each column (or choose another method)
train_df[home_features] = train_df[home_features].apply(lambda x: x.fillna(x.mean()), axis=0)
train_df[away_features] = train_df[away_features].apply(lambda x: x.fillna(x.mean()), axis=0)

# Normalize the input features
normalizer_home = Normalization(axis=-1)
normalizer_home.adapt(train_df[home_features].to_numpy())

normalizer_away = Normalization(axis=-1)
normalizer_away.adapt(train_df[away_features].to_numpy())

# Define model inputs
home_input = Input(shape=(len(home_features),), name="home_input")
away_input = Input(shape=(len(away_features),), name="away_input")

# Apply normalization
norm_home = normalizer_home(home_input)
norm_away = normalizer_away(away_input)

# Concatenate the inputs
concatenated = Concatenate()([norm_home, norm_away])

# Define the rest of the model
dense1 = Dense(128, activation='relu')(concatenated)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)

# Output layer for binary classification (Home win or not)
output_home_win = Dense(1, activation='sigmoid', name="home_win")(dense4)

# Output layer for regression (Point difference)
output_point_diff = Dense(1, name="point_diff")(dense4)

# Create the model
model = Model(inputs=[home_input, away_input], outputs=[output_home_win, output_point_diff])

# Compile the model
model.compile(optimizer='adam',
              loss={'home_win': 'binary_crossentropy', 'point_diff': 'mean_squared_error'},
              metrics={'home_win': 'accuracy', 'point_diff': 'mse'})

# Visualize the model structure
plot_model(model, show_shapes=True, show_layer_names=True, dpi=75)

# %%
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Prepare data for training
X_home = train_df[home_features].to_numpy()
X_away = train_df[away_features].to_numpy()
y_home_win = train_df['home_win'].to_numpy()
y_point_diff = train_df['point_diff'].to_numpy()

# Split the data into training and testing sets (80/20 split)
X_home_train, X_home_test, X_away_train, X_away_test, y_home_win_train, y_home_win_test, y_point_diff_train, y_point_diff_test = train_test_split(
    X_home, X_away, y_home_win, y_point_diff, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    [X_home_train, X_away_train], {'home_win': y_home_win_train, 'point_diff': y_point_diff_train},
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate the model on the test set
test_results = model.evaluate([X_home_test, X_away_test], {'home_win': y_home_win_test, 'point_diff': y_point_diff_test}, verbose=1)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['home_win_accuracy'], label='Home Win Accuracy')
    plt.plot(history.history['val_home_win_accuracy'], label='Val Home Win Accuracy')
    plt.title('Home Win Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['point_diff_mse'], label='Point Difference rmse')
    plt.plot(history.history['val_point_diff_mse'], label='Val Point Difference rmse')
    plt.title('Point Difference Mean Squared Error')
    plt.xlabel('Epochs')
    plt.ylabel('rmse')
    plt.legend()

    plt.show()

plot_history(history)

# %%



