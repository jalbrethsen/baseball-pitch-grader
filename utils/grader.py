import torch.nn as nn
import numpy as np
import torch


def get_pitcher_pitches(pitch_df, pitcher):
    """
    Filters a DataFrame of pitches to return only those thrown by a specific pitcher.

    Args:
        pitch_df (pd.DataFrame): A DataFrame containing pitch data.
        pitcher (str or int): The identifier for the pitcher to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing only the pitches thrown by the specified pitcher.
    """
    return pitch_df[pitch_df["pitcher"] == pitcher]


def grade_pitcher(
    model, pitch_df, pitcher, pitch_features, df_min, df_max, embed=False
):
    """
    Grades a pitcher's pitches using a given model.

    This function first filters the pitch DataFrame to include only pitches from the specified pitcher.
    It then preprocesses the pitch data by dropping irrelevant columns, normalizing features,
    and optionally adding 'batter' and 'pitcher' columns if embedding is enabled.
    Finally, it uses the provided model to generate predictions (grades) for the pitches
    and applies a softmax function to the output.

    Args:
        model (torch.nn.Module): The neural network model used for grading pitches.
        pitch_df (pd.DataFrame): A DataFrame containing pitch data.
        pitcher (str or int): The identifier for the pitcher to grade.
        pitch_features (list): A list of pitch features to be used as input to the model.
        df_min (pd.Series): A Series containing the minimum values for normalization.
        df_max (pd.Series): A Series containing the maximum values for normalization.
        embed (bool, optional): If True, 'batter' and 'pitcher' columns are added to the
                                 normalized input. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the softmax probabilities (grades) for each pitch.
    """
    pitch_df = get_pitcher_pitches(pitch_df, pitcher)
    X_df = pitch_df.drop(["type", "batter", "pitcher", "game_date"], axis=1)
    X_norm = (X_df - df_min) / (df_max - df_min)
    if embed:
        X_norm["batter"] = pitch_df["batter"]
        X_norm["pitcher"] = pitch_df["pitcher"]
    X_n = X_norm.to_numpy(dtype=np.float32)
    output = model(torch.tensor(X_n, device=model.device))
    output = nn.functional.softmax(output, dim=1)
    return output


def get_batter_pitches(pitch_df, batter):
    """
    Filters a DataFrame of pitches to return only those faced by a specific batter.

    Args:
        pitch_df (pd.DataFrame): A DataFrame containing pitch data.
        batter (str or int): The identifier for the batter to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing only the pitches faced by the specified batter.
    """
    return pitch_df[pitch_df["batter"] == batter]


def grade_batter(model, pitch_df, batter, pitch_features, df_min, df_max, embed=False):
    """
    Grades the pitches faced by a batter using a given model.

    This function first filters the pitch DataFrame to include only pitches faced by the specified batter.
    It then preprocesses the pitch data by dropping irrelevant columns, normalizing features,
    and optionally adding 'batter' and 'pitcher' columns if embedding is enabled.
    Finally, it uses the provided model to generate predictions (grades) for the pitches
    and applies a softmax function to the output.

    Args:
        model (torch.nn.Module): The neural network model used for grading pitches.
        pitch_df (pd.DataFrame): A DataFrame containing pitch data.
        batter (str or int): The identifier for the batter to grade.
        pitch_features (list): A list of pitch features to be used as input to the model.
        df_min (pd.Series): A Series containing the minimum values for normalization.
        df_max (pd.Series): A Series containing the maximum values for normalization.
        embed (bool, optional): If True, 'batter' and 'pitcher' columns are added to the
                                 normalized input. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the softmax probabilities (grades) for each pitch.
    """
    pitch_df = get_batter_pitches(pitch_df, batter)
    X_df = pitch_df.drop(["type", "batter", "pitcher", "game_date"], axis=1)
    X_norm = (X_df - df_min) / (df_max - df_min)
    if embed:
        X_norm["batter"] = pitch_df["batter"]
        X_norm["pitcher"] = pitch_df["pitcher"]
    X_n = X_norm.to_numpy(dtype=np.float32)
    output = model(torch.tensor(X_n, device=model.device))
    output = nn.functional.softmax(output, dim=1)
    return output
