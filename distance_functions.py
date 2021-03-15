import pandas as pd
import numpy as np


def hamming_distance(df: pd.DataFrame):
    A = df.to_numpy(int)
    return 2 * np.inner(A - .5, .5 - A) + A.shape[1] / 2
