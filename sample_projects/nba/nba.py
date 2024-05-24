# %%
import os
import pandas as pd
import nba_api.stats.endpoints
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import time
from nba_api.stats.library.parameters import *

def calculate_plus_minus(df):
    # Create a boolean column for home games
    df['HOME_GAME'] = ~df['MATCHUP'].str.contains('@')

    # Calculate the total points for each game and whether it's a home game
    game_points = df.groupby('GAME_ID')['PTS'].transform('sum')

    # Calculate the plus-minus
    df['PLUS_MINUS'] = df['PTS'] - (game_points - df['PTS'])

    #df.loc[df['HOME_GAME'], 'PLUS_MINUS'] *= -1

    return df


nba_teams = teams.get_teams()
fp_team = "./data/all_games.parquet.gzip"

if not os.path.exists(fp_team):
    print("data not found")
    all_games = pd.DataFrame()

    for team in nba_teams:
        team_id = team.get('id')
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id,league_id_nullable=LeagueIDNullable().nba, season_type_nullable=SeasonType().regular)

        team_games = gamefinder.get_data_frames()[0]
        all_games = pd.concat([all_games, team_games])
    all_games.to_parquet(fp_team, index=False)
else:
    print("data found local")
    all_games = pd.read_parquet(fp_team)

all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
all_games = all_games.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)

# Apply the function to calculate plus-minus
all_games = calculate_plus_minus(all_games)
all_games['DAYS_SINCE_LAST_GAME'] = all_games.groupby(['SEASON_ID','TEAM_ID'])['GAME_DATE'].diff().dt.days

# %%
# Franchise founding dates (replace with actual data)
franchise_founding_dates = {
    1610612742: 1980,  # Dallas Mavericks
    1610612760: 1967,  # Oklahoma City Thunder
    1610612759: 1976,  # San Antonio Spurs
    1610612765: 1948,  # Detroit Pistons
    1610612744: 1946,  # Golden State Warriors
    1610612762: 1974,  # Utah Jazz
    1610612745: 1967,  # Houston Rockets
    1610612746: 1970,  # LA Clippers
    1610612757: 1970,  # Portland Trail Blazers
    1610612758: 1948   # Sacramento Kings
}

# %%
# 1. Head-to-head records
def calculate_head_to_head(df):
    head_to_head = {}
    for _, row in df.iterrows():
        home_team = row['TEAM_ID']
        away_team = row['TEAM_ID']
        matchup = tuple(sorted([home_team, away_team]))
        
        if matchup not in head_to_head:
            head_to_head[matchup] = {'wins_home': 0, 'wins_away': 0}
        
        if row['WL'] == 'W' and not row['HOME_GAME']:
            head_to_head[matchup]['wins_away'] += 1
        elif row['WL'] == 'W' and row['HOME_GAME']:
            head_to_head[matchup]['wins_home'] += 1
    
    df['HEAD_TO_HEAD_HOME_WINS'] = df.apply(lambda row: head_to_head[tuple(sorted([row['TEAM_ID'], row['TEAM_ID']]))]['wins_home'], axis=1)
    df['HEAD_TO_HEAD_AWAY_WINS'] = df.apply(lambda row: head_to_head[tuple(sorted([row['TEAM_ID'], row['TEAM_ID']]))]['wins_away'], axis=1)

    return df

# %%
# 2. Recent performance metrics
def calculate_recent_performance(df, windows=[3, 10]):
    for window in windows:
        df[f'AVG_{window}_GAME_PERFORMANCE'] = df.groupby('TEAM_ID')['PLUS_MINUS'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        df[f'WIN_PCT_{window}_GAMES'] = df.groupby('TEAM_ID')['WL'].apply(lambda x: (x == 'W').rolling(window=window, min_periods=1).mean()).reset_index(0, drop=True)
    
    return df

# %%
# 3. Home/away splits
def calculate_home_away_splits(df):
    home_splits = df[df['HOME_GAME']].groupby('TEAM_ID')[['PLUS_MINUS', 'WL']].agg({'PLUS_MINUS': 'mean', 'WL': lambda x: (x == 'W').mean()}).reset_index()
    home_splits = home_splits.rename(columns={'PLUS_MINUS': 'HOME_PLUS_MINUS', 'WL': 'HOME_WIN_PCT'})
    
    away_splits = df[~df['HOME_GAME']].groupby('TEAM_ID')[['PLUS_MINUS', 'WL']].agg({'PLUS_MINUS': 'mean', 'WL': lambda x: (x == 'W').mean()}).reset_index()
    away_splits = away_splits.rename(columns={'PLUS_MINUS': 'AWAY_PLUS_MINUS', 'WL': 'AWAY_WIN_PCT'})
    
    df = pd.merge(df, home_splits, on='TEAM_ID', how='left')
    df = pd.merge(df, away_splits, on='TEAM_ID', how='left')
    
    return df

# %%
# 4. Rest days since last game
def calculate_rest_days(df):
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    df['REST_DAYS'].fillna(0, inplace=True)
    return df

# %%
# 5. Franchise age
def calculate_franchise_age(df, franchise_founding_dates):
    df['FRANCHISE_AGE'] = df['GAME_DATE'].dt.year - df['TEAM_ID'].map(franchise_founding_dates)
    return df

# %%
# 6. Cumulative season performance
def calculate_cumulative_season_performance(df):
    df['CUMULATIVE_SEASON_PLUS_MINUS'] = df.groupby(['SEASON_ID', 'TEAM_ID'])['PLUS_MINUS'].cumsum()
    df['CUMULATIVE_SEASON_WINS'] = df.groupby(['SEASON_ID', 'TEAM_ID'])['WL'].apply(lambda x: (x == 'W').cumsum())
    df['CUMULATIVE_SEASON_LOSSES'] = df.groupby(['SEASON_ID', 'TEAM_ID'])['WL'].apply(lambda x: (x == 'L').cumsum())
    return df

# %%
# 7. Previous season performance
def calculate_previous_season_performance(df):
    previous_season_performance = df.groupby(['SEASON_ID', 'TEAM_ID'])[['PLUS_MINUS', 'WL']].sum().reset_index()
    previous_season_performance['SEASON_ID'] = previous_season_performance['SEASON_ID'] + 1
    previous_season_performance = previous_season_performance.rename(columns={'PLUS_MINUS': 'PREVIOUS_SEASON_PLUS_MINUS', 'WL': 'PREVIOUS_SEASON_WINS'})
    df = pd.merge(df, previous_season_performance, on=['SEASON_ID', 'TEAM_ID'], how='left')
    return df

# %%
# Apply the functions to create new features
all_games = calculate_head_to_head(all_games)
all_games = calculate_recent_performance(all_games)
all_games = calculate_home_away_splits(all_games)
all_games = calculate_rest_days(all_games)
all_games = calculate_franchise_age(all_games, franchise_founding_dates)
all_games = calculate_cumulative_season_performance(all_games)
all_games = calculate_previous_season_performance(all_games)

# %%
# Save the updated dataframe to a new file
all_games.to_parquet("./data/all_games_with_features.parquet.gzip", index=False)