import pandas as pd
from estimators import IPW, CovariateAdjustment, Matching
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from distance_functions import hamming_distance
import numpy as np

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
    for data in ["data1.csv", "data2.csv"]:


        print(" ----------  Working on", data)
        att = []
        data1 = pd.read_csv(data)
        x1, y1 = preprocess_data(data1)



        # ------ IPW calculation -------------
        l = IPW()
        attIPW1, propScore = l.estimate(x1, y1)
        print(f"{l.name}: {attIPW1}")
        # print(propScore[:,0])
        propList.append([data] + list(propScore[:,0]))
        att.append(attIPW1)

        # ------ S learner calculation -------------

        l = CovariateAdjustment(learner='s')
        attS = l.estimate(x1, y1)
        print(f"{l.name}: {attS}")
        att.append(attS)


        # ------ T learner calculation -------------

        l = CovariateAdjustment(learner='t')
        attT = l.estimate(x1, y1)
        print(f"{l.name}: {attT}")
        att.append(attT)


        # ------ Matching calculation  -------------

        l = Matching(euclidean_distances)
        attME = l.estimate(x1, y1)
        print(f"{l.name}: {attME}")
        att.append(attME)


        l = Matching(manhattan_distances)
        attM = l.estimate(x1, y1)
        print(f"{l.name}: {attM}")

        l = Matching(hamming_distance)
        attH = l.estimate(x1, y1)
        print(f"{l.name}: {attH}")

        # ------ best estimate of ATT via averaging -------------

        IPWavg = []
        for k in range(0,30):
            l = IPW()
            attBest, propScore = l.estimate(x1, y1)
            IPWavg.append(attBest)

        IPWavg= np.asarray(IPWavg).mean()
        attBest = np.asarray([IPWavg,attS,attT,attM,attH,attME]).mean()
        print("best ATT",attBest)
        att.append(attBest)

        attList.append(att)



    # ------ save all the results -------------

    propList = pd.DataFrame(propList)
    propList.to_csv("models_propensity.csv", header=False, index=False)

    attList = pd.DataFrame(attList).T
    attList.columns = ['data1','data2']
    type = [1, 2, 3, 4, 5]
    attList.insert(loc=0, column='Type', value=type)
    print (attList)
    attList.to_csv("ATT_results.csv",index=False)


