import os
import pandas as pd
import nba_api.stats.endpoints
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import time
from nba_api.stats.library.parameters import *


def calculate_plus_minus(df):
    df["HOME_GAME"] = ~df["MATCHUP"].str.contains("@")
    game_points = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["PLUS_MINUS"] = df["PTS"] - (game_points - df["PTS"])
    return df

def calculate_recent_performance(df, windows=[3, 10], cols=None):
    if cols is None:
        cols = [
            "PLUS_MINUS",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
        ]

    for window in windows:
        for col in cols:
            df[f"AVG_{window}_GAME_{col}"] = (
                df.groupby("TEAM_ID")[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)  # Shift the result by 1 to exclude the current game
                .reset_index(0, drop=True)
            )

        df[f"WIN_PCT_{window}_GAMES"] = (
            df.groupby("TEAM_ID")["WL"]
            .apply(lambda x: (x == "W").rolling(window=window, min_periods=1).mean())
            .shift(1)  # Shift the result by 1 to exclude the current game
            .reset_index(0, drop=True)
        )
    return df