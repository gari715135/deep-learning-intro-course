# %%
import os
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import time
from nba_api.stats.library.parameters import *

nba_teams = teams.get_teams()
fp_team = "./data/all_games.parquet.gzip"

if not os.path.exists(fp_team):
    print("data not found")
    all_games = pd.DataFrame()

    for team in nba_teams:
        team_id = team.get("id")
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            league_id_nullable=LeagueIDNullable().nba,
            season_type_nullable=SeasonType().regular,
        )

        team_games = gamefinder.get_data_frames()[0]
        all_games = pd.concat([all_games, team_games])
    all_games.to_parquet(fp_team, index=False)
else:
    print("data found local")
    all_games = pd.read_parquet(fp_team)

franchise_founding_dates = {team["id"]: team["year_founded"] for team in nba_teams}

all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])
all_games = all_games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

# %%
def calculate_plus_minus(df):
    df["HOME_GAME"] = ~df["MATCHUP"].str.contains("@")
    game_points = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["PLUS_MINUS"] = df["PTS"] - (game_points - df["PTS"])
    return df


def calculate_head_to_head(df):
    head_to_head = {}
    for _, row in df.iterrows():
        home_team = row["TEAM_ID"]
        away_team = row["TEAM_ID"]
        matchup = tuple(sorted([home_team, away_team]))

        if matchup not in head_to_head:
            head_to_head[matchup] = {"wins_home": 0, "wins_away": 0}

        if row["WL"] == "W" and not row["HOME_GAME"]:
            head_to_head[matchup]["wins_away"] += 1
        elif row["WL"] == "W" and row["HOME_GAME"]:
            head_to_head[matchup]["wins_home"] += 1

    df["HEAD_TO_HEAD_HOME_WINS"] = df.apply(
        lambda row: head_to_head[tuple(sorted([row["TEAM_ID"], row["TEAM_ID"]]))][
            "wins_home"
        ],
        axis=1,
    )
    df["HEAD_TO_HEAD_AWAY_WINS"] = df.apply(
        lambda row: head_to_head[tuple(sorted([row["TEAM_ID"], row["TEAM_ID"]]))][
            "wins_away"
        ],
        axis=1,
    )

    return df


def calculate_home_away_splits(df):
    home_splits = (
        df[df["HOME_GAME"]]
        .groupby("TEAM_ID")[["PLUS_MINUS", "WL"]]
        .agg({"PLUS_MINUS": "mean", "WL": lambda x: (x == "W").mean()})
        .reset_index()
    )
    home_splits = home_splits.rename(
        columns={"PLUS_MINUS": "HOME_PLUS_MINUS", "WL": "HOME_WIN_PCT"}
    )

    away_splits = (
        df[~df["HOME_GAME"]]
        .groupby("TEAM_ID")[["PLUS_MINUS", "WL"]]
        .agg({"PLUS_MINUS": "mean", "WL": lambda x: (x == "W").mean()})
        .reset_index()
    )
    away_splits = away_splits.rename(
        columns={"PLUS_MINUS": "AWAY_PLUS_MINUS", "WL": "AWAY_WIN_PCT"}
    )

    df = pd.merge(df, home_splits, on="TEAM_ID", how="left")
    df = pd.merge(df, away_splits, on="TEAM_ID", how="left")

    return df


def calculate_recent_performance(df, windows=[3, 10]):
    for window in windows:
        df[f"AVG_{window}_GAME_PERFORMANCE"] = (
            df.groupby("TEAM_ID")["PLUS_MINUS"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df[f"WIN_PCT_{window}_GAMES"] = (
            df.groupby("TEAM_ID")["WL"]
            .apply(lambda x: (x == "W").rolling(window=window, min_periods=1).mean())
            .reset_index(0, drop=True)
        )
    return df


def calculate_rest_days(df):
    df["REST_DAYS"] = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    df["REST_DAYS"] = df["REST_DAYS"].fillna(0)
    return df


def calculate_franchise_age(df, franchise_founding_dates):
    df["FRANCHISE_AGE"] = df["GAME_DATE"].dt.year - df["TEAM_ID"].map(
        franchise_founding_dates
    )
    return df


def calculate_cumulative_season_performance(df):
    # Calculate the cumulative sum for PLUS_MINUS
    df["CUMULATIVE_SEASON_PLUS_MINUS"] = df.groupby(["SEASON_ID", "TEAM_ID"])[
        "PLUS_MINUS"
    ].cumsum()

    # Calculate the cumulative sum for WINS
    df["CUMULATIVE_SEASON_WINS"] = df.groupby(["SEASON_ID", "TEAM_ID"])["WL"].transform(
        lambda x: (x == "W").cumsum()
    )

    # Calculate the cumulative sum for LOSSES
    df["CUMULATIVE_SEASON_LOSSES"] = df.groupby(["SEASON_ID", "TEAM_ID"])[
        "WL"
    ].transform(lambda x: (x == "L").cumsum())

    return df

# %%
# Apply the function to calculate plus-minus
all_games = calculate_plus_minus(all_games)
all_games = calculate_head_to_head(all_games)
all_games = calculate_recent_performance(all_games, windows=[3, 10])
all_games = calculate_home_away_splits(all_games)
all_games = calculate_rest_days(all_games)
all_games = calculate_franchise_age(all_games, franchise_founding_dates)
all_games = calculate_cumulative_season_performance(all_games)
