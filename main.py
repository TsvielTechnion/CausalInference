import pandas as pd

from read_NSDUH import Read, Codes
from estimators_ATE import IPW, CovariateAdjustment, Matching
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from distance_functions import hamming_distance
from read_NSDUH import Codes

OVERLAP_PCT = .95

if __name__ == "__main__":

    propList = []
    attList = []
    results = []
    for year in ["2017", "2018", "2019"]:
        file = f"NSDUH_{year}.parquet"
        reader = Read(file_path=file)
        data = reader.read().loc[:, :].reset_index(drop=True)

        x = data.loc[:, ~data.columns.isin(Codes.outcomes.values())]
        ipw = IPW(x=x)
        p_scores = pd.DataFrame(ipw.p_scores)
        only_overlapped_data = p_scores[(p_scores[0] < OVERLAP_PCT) & (p_scores[1] < OVERLAP_PCT)]

        x1 = x.loc[only_overlapped_data.index]
        y = data.loc[only_overlapped_data.index]

        unfiltered_methods = [
            # Matching(hamming_distance, list(set(x1.columns) - set(Codes.non_psy_drugs.values()) - set(['T']))),
            # Matching(hamming_distance, Codes.non_psy_drugs.values()),
            Matching(hamming_distance)

        ]
        filtered_methods = [
            CovariateAdjustment(learner='s'),
            CovariateAdjustment(learner='t'),
        ]

        T1_overlap = x1[x1['T'] == 1].shape[0]
        T0_overlap = x1[x1['T'] == 0].shape[0]

        T1_matching = x[x['T'] == 1].shape[0]
        T0_matching = x[x['T'] == 0].shape[0]

        for outcome in Codes.outcomes.values():
            print(f"------{outcome}------")
            y1 = y[outcome]
            for method in filtered_methods:
                mean, std = method.estimate(x1, y1)
                results.append([method.name, mean, std, T1_overlap, T0_overlap, year])
                print({method.name: {
                    "mean": round(mean, 3),
                    "std": round(std, 3)
                }})

            x0 = x
            y0 = data[outcome]
            for method in unfiltered_methods:
                mean, std = method.estimate(x0, y0)
                results.append([method.name, mean, std, T1_matching, T0_matching, year])
                print({method.name: {
                    "mean": round(mean, 3),
                    "std": round(std, 3)
                }})

            print(f"---------------------------")
    results = pd.DataFrame(results, columns=["Estimator", "Mean", "STD", "T1", "T0", "Year"])
    results.to_csv(f"results.csv")
