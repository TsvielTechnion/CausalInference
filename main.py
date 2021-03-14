import pandas as pd

from read_NSDUH import Read, Codes
from estimators import IPW, CovariateAdjustment, Matching
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

OVERLAP_PCT = .9


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

    x = data.loc[:, ~data.columns.isin(Codes.outcomes.values())]
    ipw = IPW(x=x)
    p_scores = pd.DataFrame(ipw.p_scores)
    only_overlapped_data = p_scores[(p_scores[0] < OVERLAP_PCT) & (p_scores[1] < OVERLAP_PCT)]

    x1 = x.loc[only_overlapped_data.index].reset_index()
    y = data.loc[only_overlapped_data.index].reset_index()

    filtered_methods = [
        CovariateAdjustment(learner='s'),
        CovariateAdjustment(learner='t'),
    ]
    unfiltered_methods = [
        Matching(euclidean_distances),
        Matching(manhattan_distances),
    ]

    for outcome in Codes.outcomes.values():
        print(f"------{outcome}------")
        y1 = y[outcome]
        for method in filtered_methods:
            mean, std = method.estimate(x1, y1)
            print({method.name: {
                "mean": round(mean, 2),
                "std": std
            }})

        x0 = x
        y0 = data[outcome]
        for method in unfiltered_methods:
            mean, std = method.estimate(x0, y0)
            print({method.name: {
                "mean": round(mean, 2),
                "std": std
            }})

        print(f"---------------------------")
