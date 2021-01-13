import pandas as pd
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
    data1 = pd.read_csv("data1.csv")
    data2 = pd.read_csv("data2.csv")
    x1, y1 = preprocess_data(data1)
    x2, y2 = preprocess_data(data2)

    learners = [
                CovariateAdjustment(learner='s'),
                CovariateAdjustment(learner='t'),
                IPW(),
                Matching(manhattan_distances),
                Matching(euclidean_distances),
                Matching(hamming_distance)
                ]
    for l in learners:
        att = l.estimate(x1, y1)
        print(f"{l.name}: {att}")