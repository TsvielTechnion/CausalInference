import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import deque
from itertools import product


class Codes:
    code2name = {
        "AGE2": "AGE",
        "HEALTH2": "OVERALL_HEALTH",
        "SEXATRACT": "SEXUAL_ATTRACTION",
        "IRSEX": "M/F",
        "IREDUHIGHST2": "EDUCATION",
        "NEWRACE2": "RACE",
        # "PRVHLTIN": "COVERED_PRIVATE_INSURANCE",
        "INCOME": "TOTAL_FAMILY_INCOME"
    }


class Read:
    def __init__(self, file_path):
        self.file = file_path

    def read(self):
        df = pd.read_parquet(self.file)[Codes.code2name.keys()]
        df = df.rename(columns=Codes.code2name)
        return df


if __name__ == "__main__":
    reader = Read(file_path="NSDUH_2019.parquet")
    df = reader.read()
    df.fillna(-1, inplace=True)

    number_of_figs = Codes.code2name.values().__len__()
    dim = math.ceil(number_of_figs ** 0.5)
    fig, ax = plt.subplots(dim, dim)

    comb = product(range(dim), range(dim))
    d = deque(comb)

    for col in sorted(df.columns):
        x, y = d.popleft()
        data = df[col].astype(int)
        sns.histplot(data, ax=ax[x][y])

    plt.show()
