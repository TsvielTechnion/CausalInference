import pandas as pd
from estimators import IPW, CovariateAdjustment
import seaborn as sns
from sklearn import preprocessing


def preprocess_data(data: pd.DataFrame):
    # Remove sample id column
    data.drop(columns='Unnamed: 0', inplace=True)
    # categorize feature x_2
    data['x_2'] = data['x_2'].astype('category').cat.codes
    data['x_24'] = data['x_24'].astype('category').cat.codes
    data['x_21'] = data['x_21'].astype('category').cat.codes

    y = data['Y']
    x = data.loc[:, data.columns != 'Y']
    return x, y


if __name__ == "__main__":
    data1 = pd.read_csv("data1.csv")
    data2 = pd.read_csv("data2.csv")
    x1, y1 = preprocess_data(data1)
    x2, y2 = preprocess_data(data2)

    s_learner = CovariateAdjustment(learner='s').estimate(x1, y1)
    t_learner = CovariateAdjustment(learner='t').estimate(x1, y1)
    ATT_IPW = IPW().estimate(x1, y1)

    print()