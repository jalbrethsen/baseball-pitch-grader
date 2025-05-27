import pandas as pd
import numpy as np
from pathlib import Path
from utils.text_mappings import outcomes_v2, pitch_types, righty_lefty, pitch_outcomes
from pybaseball import statcast, cache
import json


def read_from_cache(columns=None, directory="~/.pybaseball/cache/"):
    """
    For duplicate queries we read from cache on disk
    """
    data_dir = Path(directory)
    parquets = []
    for json_file in data_dir.glob("get_statcast_data_from_csv*.cache_record.json"):
        print(json_file)
        with open(json_file, "r") as jsonin:
            metadata = json.load(jsonin)
        if metadata["func"] == "get_statcast_data_from_csv_url":
            parquets.append(metadata["dataframe"])
    full_df = pd.concat(
        pd.read_parquet(parquet_file, columns=columns) for parquet_file in parquets
    )
    return full_df


def pull_statcast_year(year, columns=None, verbose=False):
    """
    For single year we pull statcast data and optionally return a subset of columns.
    Purpose of function is to allow us to pull smaller chunks and filter out unused columns.
    """
    start_dt = f"{year}-01-01"
    end_dt = f"{int(year)+1}-01-01"
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    if verbose:
        print("Finished year ", year)
    if columns:
        return df[columns]
    return df


def pull_statcast_multiyear(start_year, end_year, columns=None, verbose=False):
    """
    We pull a year at a time and filter out unused columns, this helps us prevent using all our RAM
    """
    full_df = pd.concat(
        pull_statcast_year(year, columns=columns, verbose=verbose)
        for year in range(int(start_year), int(end_year) + 1)
    )
    return full_df


def get_valid_batters(df, min_pitches):
    """
    This function filters out the batters who are less than the minimum atbats
    """
    valid_batters = df.groupby("batter").count()["game_pk"] >= min_pitches
    valid_batter_ids = valid_batters[valid_batters].index
    df = df[df["batter"].isin(valid_batter_ids)]
    # map batter id to embedding index
    batter_map = {int(valid_batter_ids[i]): i for i in range(len(valid_batter_ids))}
    df.batter = df.batter.map(batter_map)
    return df, batter_map


def get_valid_pitchers(df, min_pitches):
    """
    This function filters out the pitchers who are less than the minimum atbats
    """
    valid_pitchers = df.groupby("pitcher").count()["game_pk"] >= min_pitches
    valid_pitcher_ids = valid_pitchers[valid_pitchers].index
    df = df[df["pitcher"].isin(valid_pitcher_ids)]
    # map pitcher id to embedding index
    pitcher_map = {int(valid_pitcher_ids[i]): i for i in range(len(valid_pitcher_ids))}
    df.pitcher = df.pitcher.map(pitcher_map)
    return df, pitcher_map


def get_prev_pitch_features(df):
    df = df.sort_values(by=["game_pk", "at_bat_number", "pitch_number"])
    df["previous_pitch_speed"] = df.groupby(["game_pk", "at_bat_number"])[
        "release_speed"
    ].shift(1)
    df["previous_pitch_type"] = df.groupby(["game_pk", "at_bat_number"])[
        "pitch_type"
    ].shift(1)
    df["previous_zone"] = df.groupby(["game_pk", "at_bat_number"])["zone"].shift(1)
    df["previous_plate_x"] = df.groupby(["game_pk", "at_bat_number"])["plate_x"].shift(
        1
    )
    df["previous_plate_z"] = df.groupby(["game_pk", "at_bat_number"])["plate_z"].shift(
        1
    )
    return df


def preprocess(df, pitch_features, min_pitches=5000):
    """
    Take input dataframe of statcast data
    Get whether the ball is ball, strike, or inplay mapped to integer
    Clean up valid pitches and regular season games
    """
    # get regular season games
    df = df[df.game_type == "R"]
    df = get_prev_pitch_features(df)
    df["type"] = df["type"].map(pitch_outcomes)
    df, pitcher_map = get_valid_pitchers(df, min_pitches)
    df, batter_map = get_valid_batters(df, min_pitches)
    # filter out unneeded features to reduce processing time
    df = df[pitch_features]
    # filter out unknown pitch types
    df.fillna(0, inplace=True)
    return df, batter_map, pitcher_map
