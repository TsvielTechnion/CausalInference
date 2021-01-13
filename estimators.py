import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Estimator:
    def estimate(self, x, y) -> int:
        ...


class IPW(Estimator):
    def estimate(self, x:pd.DataFrame, y: pd.DataFrame):
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
        classifier = RandomForestClassifier().fit(X=features, y=t)
        return classifier.predict_proba(features)
