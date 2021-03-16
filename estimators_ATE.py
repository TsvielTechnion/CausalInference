import pandas as pd
import numpy as np
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV


class Estimator:
    def estimate(self, x: pd.DataFrame, y: pd.DataFrame) -> int:
        ...


class IPW(Estimator):
    name = "IPW"

    def __init__(self, x: pd.DataFrame):
        self.p_scores = self.estimate_propensity(x)

    def estimate(self, x: pd.DataFrame, y: pd.DataFrame) -> int:
        p_scores = self.p_scores[x.index]
        t = x['T'].to_numpy()
        y = y.to_numpy()
        p_scores_ratio = p_scores[:, 1] / p_scores[:, 0]
        sigma_T = t.sum()
        sigma_ti_yi = (t * y).sum()
        sigma_minus_ti_y1 = ((1 - t) * y * p_scores_ratio).sum()
        sigma_minus_ti = ((1 - t) * p_scores_ratio).sum()
        return (sigma_ti_yi / sigma_T) / (sigma_minus_ti_y1 / sigma_minus_ti), -1

    def estimate_propensity(self, x: pd.DataFrame):
        trees = 90
        depth = 15
        t = x['T'].to_numpy()
        features = x.loc[:, x.columns != 'T'].to_numpy()
        # classifier = RandomForestClassifier(n_estimators=trees, max_depth=depth).fit(X=features, y=t)
        classifier = LogisticRegressionCV(max_iter=10_000).fit(X=features, y=t)
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
        columns = list(x.columns)
        columns.remove('T')

        x_t1 = x[x['T'] == 1]
        x_t1_0 = copy.deepcopy(x_t1)
        x_t1_0['T'] = 0

        features = x.to_numpy()
        y_all = y.to_numpy()
        predictor = LogisticRegressionCV(max_iter=10_000).fit(X=features, y=y_all)

        y_hat_0 = predictor.predict_proba(x_t1_0)[:, 1]
        y_hat_1 = predictor.predict_proba(x_t1)[:, 1]

        ATE = y_hat_1 - y_hat_0
        return ATE.mean(), ATE.std()

    def t_learner(self, x:pd.DataFrame, y: pd.DataFrame) -> int:
        x_t1 = x[x['T'] == 1]
        y_t1 = y[x_t1.index]
        x_t0 = x[x['T'] == 0]
        y_t0 = y[x_t0.index]

        x_t1_0 = copy.deepcopy(x_t1)
        x_t1_0['T'] = 0

        predictor0 = LogisticRegressionCV(max_iter=10_000).fit(X=x_t0.to_numpy(), y=y_t0)
        predictor1 = LogisticRegressionCV(max_iter=10_000).fit(X=x_t1.to_numpy(), y=y_t1)

        y_hat_0 = predictor0.predict_proba(x_t1_0)[:, 1]
        y_hat_1 = predictor1.predict_proba(x_t1)[:, 1]

        ATE = y_hat_1 - y_hat_0
        return ATE.mean(), ATE.std()


class Matching(Estimator):
    name = "Matching 1-20"

    def __init__(self, distance_function, match_subset=None):
        self.distance_function = distance_function
        self._dis_mat = None
        self._balance = None
        self.match_subset = match_subset

    def estimate(self, x: pd.DataFrame, y: pd.DataFrame) -> int:
        predictor = LogisticRegressionCV(max_iter=10_000).fit(X=x, y=y)
        y = predictor.predict_proba(x)
        couples = self.match(x)
        # self._calc_balance(couples=couples, x=x)

        ATE = []
        for t1, t0_couple in couples.items():
            ATE.append((y[t1][1] - y[t0_couple][:, 1].mean()))

        ATE = np.array(ATE)
        return ATE.mean(), ATE.std()

    def match(self, x: pd.DataFrame):
        t0_indices = x[x['T'] == 0].index
        t1_indices = x[x['T'] == 1].index

        x_without_t = x.loc[:, x.columns != 'T']
        distances_df = pd.DataFrame(self.distance_matrix(x_without_t)).loc[t1_indices, t0_indices]

        t12group = {}
        for i in distances_df.index:
            r = distances_df.loc[i, :]
            idx = r.sort_values()[:20]
            t12group[i] = idx.index
        # couples = distances_df.idxmin(axis=1)
        return t12group

    def distance_matrix(self, x):
        if self._dis_mat is None:
            x_subset = x[self.match_subset] if self.match_subset else x
            self._dis_mat = self.distance_function(x_subset)
        return self._dis_mat

    def _calc_balance(self, couples, x):
        if self._balance is not None:
            return
        t1_idx = list(couples.index)
        t0_idx = list(couples.values)
        df_t1 = x.loc[t1_idx]
        df_t0 = x.loc[t0_idx]
        balance = []
        for c in df_t0.columns:
            balance.append(df_t0[c].value_counts())
            balance.append(df_t1[c].value_counts())

        balance_df = pd.DataFrame(balance)
        balance_df = balance_df / balance_df.sum(axis=1)[:, None]
        balance_df.to_csv(f"Balance_{self.name}_{hash(str(self.match_subset))}.csv")
        self._balance = 1