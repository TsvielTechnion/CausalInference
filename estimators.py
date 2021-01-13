import pandas as pd
import numpy as np
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


class Estimator:
    def estimate(self, x:pd.DataFrame, y: pd.DataFrame) -> int:
        ...


class IPW(Estimator):
    name = "IPW"

    def estimate(self, x:pd.DataFrame, y: pd.DataFrame) -> int:
        p_scores = self.estimate_propensity(x)
        t = x['T'].to_numpy()
        y = y.to_numpy()
        p_scores_ratio = p_scores[:, 1] / p_scores[:, 0]

        sigma_T = t.sum()
        sigma_ti_yi = (t * y).sum()
        sigma_minus_ti_y1 = ((1 - t) * y * p_scores_ratio).sum()
        sigma_minus_ti = ((1 - t) * p_scores_ratio).sum()
        return sigma_ti_yi / sigma_T - sigma_minus_ti_y1 / sigma_minus_ti

    def estimate_propensity(self, x: pd.DataFrame):
        t = x['T'].to_numpy()
        features = x.loc[:, x.columns != 'T'].to_numpy()
        classifier = RandomForestClassifier(n_estimators=20, max_depth=5).fit(X=features, y=t)
        return classifier.predict_proba(features)


class CovariateAdjustment(Estimator):
    name = "_Learner"

    def __init__(self, learner: str):
        if learner == "s":
            self.estimate = self.s_learner
        elif learner == "t":
            self.estimate = self.t_learner
        else:
            raise Exception("Unknown learner")

        self.name = learner + self.name

    def estimate(self, x: pd.DataFrame, y: pd.DataFrame) -> int:
        ...

    def s_learner(self, x: pd.DataFrame, y: pd.DataFrame) -> int:
        x_t1 = x[x['T'] == 1]

        x_t1_0 = copy.deepcopy(x_t1)
        x_t1_0['T'] = 0

        features = x.to_numpy()
        y_all = y.to_numpy()

        predictor = LinearRegression(normalize=True).fit(X=features, y=y_all)
        y_hat_0 = predictor.predict(x_t1_0)
        y_hat_1 = predictor.predict(x_t1)
        return (y_hat_1 - y_hat_0).mean()

    def t_learner(self, x:pd.DataFrame, y: pd.DataFrame) -> int:
        x_t1 = x[x['T'] == 1]
        y_t1 = y[x_t1.index]
        x_t0 = x[x['T'] == 0]
        y_t0 = y[x_t0.index]

        x_t1_0 = copy.deepcopy(x_t1)
        x_t1_0['T'] = 0

        predictor0 = LinearRegression(normalize=True).fit(X=x_t0.to_numpy(), y=y_t0)
        predictor1 = LinearRegression(normalize=True).fit(X=x_t1.to_numpy(), y=y_t1)

        y_hat_0 = predictor0.predict(x_t1_0)
        y_hat_1 = predictor1.predict(x_t1)
        return (y_hat_1 - y_hat_0).mean()