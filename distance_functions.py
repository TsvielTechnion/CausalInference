import pandas as pd


def hamming_distance(df: pd.DataFrame):
    A = df.values
    return (A[:, None, :] != A).sum(2)
