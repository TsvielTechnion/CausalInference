import pandas as pd
import numpy as np

from read_NSDUH import Read, Codes
from estimators import IPW, CovariateAdjustment, Matching
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from distance_functions import hamming_distance


def preprocess_data(data: pd.DataFrame):
    # Remove sample id column
    data.drop(columns='Unnamed: 0', inplace=True)
    # categorize features
    data = pd.get_dummies(data)
    y = data['Y']
    x = data.loc[:, data.columns != 'Y']
    return x, y


if __name__ == "__main__":

    propList = []
    attList = []
    reader = Read(file_path="NSDUH_2019.parquet")
    data = reader.read().loc[:, :].reset_index(drop=True)

    x1 = data.loc[:, ~data.columns.isin(Codes.outcomes.values())]

    methods = [IPW(),
               CovariateAdjustment(learner='s'),
               CovariateAdjustment(learner='t'),
               Matching(euclidean_distances),
               Matching(manhattan_distances),
               Matching(hamming_distance),
               ]

    for outcome in Codes.outcomes.values():
        print(f"------{outcome}------")
        y1 = data[outcome]
        for method in methods:
            result = method.estimate(x1, y1)
            print({method.name: f"{round(100 * result, 2)}%"})
        print(f"---------------------------")
