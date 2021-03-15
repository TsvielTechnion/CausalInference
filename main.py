import pandas as pd

from read_NSDUH import Read, Codes
from estimators import IPW, CovariateAdjustment, Matching
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from distance_functions import hamming_distance
from read_NSDUH import Codes

OVERLAP_PCT = 95


if __name__ == "__main__":

    propList = []
    attList = []
    reader = Read(file_path="NSDUH_2019.parquet")
    data = reader.read().loc[:, :].reset_index(drop=True)

    x = data.loc[:, ~data.columns.isin(Codes.outcomes.values())]
    ipw = IPW(x=x)
    p_scores = pd.DataFrame(ipw.p_scores)
    only_overlapped_data = p_scores[(p_scores[0] < OVERLAP_PCT) & (p_scores[1] < OVERLAP_PCT)]

    x1 = x.loc[only_overlapped_data.index]
    y = data.loc[only_overlapped_data.index]

    filtered_methods = [
        CovariateAdjustment(learner='s'),
        CovariateAdjustment(learner='t'),
    ]
    unfiltered_methods = [
        Matching(hamming_distance, Codes.non_psy_drugs.values()),
        Matching(hamming_distance)

    ]

    for outcome in Codes.outcomes.values():
        print(f"------{outcome}------")
        y1 = y[outcome]
        for method in filtered_methods:
            mean, std = method.estimate(x1, y1)
            print({method.name: {
                "mean": round(mean, 3),
                "std": round(std, 3)
            }})

        x0 = x
        y0 = data[outcome]
        for method in unfiltered_methods:
            mean, std = method.estimate(x0, y0)
            print({method.name: {
                "mean": round(mean, 3),
                "std": round(std, 3)
            }})

        print(f"---------------------------")
