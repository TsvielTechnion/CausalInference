import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import deque
from itertools import product
from sklearn.impute import SimpleImputer


class Const:
    code2minage = {
        1: 12,
        2: 13,
        3: 14,
        4: 15,
        5: 16,
        6: 17,
        7: 18,
        8: 19,
        9: 20,
        10: 21,
        11: 22,
        12: 24,
        13: 26,
        14: 30,
        15: 35,
        16: 50,
        17: 65
    }

    categorical = {"SEXIDENT",
                   "SEXATRACT",
                   "NEWRACE2",
                   "IRSEX",
                   "HEALTH2",
                   "CATAG6",
                   "INCOME",
                   "IREDUHIGHST2",
                   "COUTYP4",
                   "WRKNUMJOB2",
                   "RSKYFQTES"}


class Codes:
    demographics = {
        "AGE2": "AGE",
        "HEALTH2": "OVERALL HEALTH",
        "SEXATRACT": "SEXUAL ATTRACTION",
        "SEXIDENT": "SEXUAL IDENTITY",
        "IRSEX": "M/F",
        "NEWRACE2": "RACE",
        "CATAG6": "Categorical AGE",
        "INCOME": "TOTAL FAMILY INCOME",

        "IREDUHIGHST2": "EDUCATION",
        "COUTYP4": "City/SCity/NonCity",
        "WRKNUMJOB2": "PAST 12 MOS, HOW MANY EMPLOYERS",

        "HALLUCAGE": "Age first used",
        # "IRTRQANYREC": "Tranq recent"
    }

    personality = {
        "RSKYFQTES": "LIKE TO TEST YOURSELF BY DOING RISKY THINGS",
    }

    psy_drugs = {
        "PSILCY2": "EVER PSILOCYBIN",
        "MESC2": "EVER MESCALIN",
        "PEYOTE2": "EVER PEYOTE",
        "LSDFLAG": "EVER LSD",
        "DAMTFXFLAG": "EVER DMT"
    }

    outcomes = {
        "SPDMON": "PAST MONTH SERIOUS DISTRESS",
        "MHSUITHK": "SERIOUSLY THOUGHT ABOUT KILLING SELF IN PAST YEAR",
        "MHSUITRY": "ATTEMPTED TO KILL SELF IN PAST YEAR",
        "MHSUIPLN": "MADE PLANS TO KILL SELF IN PAST YEAR"
    }

    non_psy_drugs = {
        "COCFLAG": "cocaine",
        "CRKFLAG": "crack",
        "HERFLAG": "heroin",
        "CIGFLAG": "cigarettes",
        "CGRFLAG": "cigars",
        "PIPFLAG": "pipe",
        "SMKLSSFLAG": "smokeless tobacco",
        "TOBFLAG": "any tobacco",
        "ALCFLAG": "alcohol",
        "PCPFLAG": "PCP",
        "PNRANYFLAG": "pain reliever",
        "METHAMFLAG": "METHAMPHETAMINE",
        "INHALFLAG": "RC-INHALANTS - EVER USED",
        "STMANYFLAG": "stimulants",

        # "MRJFLAG": "marujuana",
        # "ECSTMOFLAG": "ecstasy",
        # "KETMINFLAG": "ketamine",
        # "SALVIAFLAG": "salvia",
        # "TRQANYFLAG": "tranquilizers",
        # "SEDANYFLAG": "sedatives",
        # "PSYANYFLAG": "psychotherapuetics"

    }

    all_codes = {
        **demographics,
        **personality,
        **psy_drugs,
        **non_psy_drugs,
        **outcomes
    }


class Read:
    def __init__(self, file_path):
        self.file = file_path

    def read(self):
        df = pd.read_parquet(self.file)[Codes.all_codes.keys()]
        df["AGE2"].replace(Const.code2minage, inplace=True)

        # Treatment is used psychodelics & and first time use isn't wasn't the last year.
        df["T"] = ((df["PSILCY2"] > 0) | (df["MESC2"] > 0) | (df["PEYOTE2"] > 0) | (df["LSDFLAG"] > 0) | (
                    df["DAMTFXFLAG"] > 0)) & (df['AGE2'] > df["HALLUCAGE"])
        df["T"] = df['T'].astype(int)

        # Remove not answered behavior questions or last year use of psy
        df = df[(df['RSKYFQTES'] < 80) & (df['SEXIDENT'] < 80) & (df['SEXATRACT'] < 80) & (df['WRKNUMJOB2'] < 80) &
                (df['AGE2'] != df["HALLUCAGE"])]

        df.drop(columns=[*Codes.psy_drugs.keys(),
                         "HALLUCAGE",
                         "AGE2"], inplace=True)

        # Categorical columns should be zero one encoded
        for i in Const.categorical:
            df[i] = df[i].astype(str)
        df = pd.get_dummies(df)
        df = df.rename(columns=Codes.all_codes)

        df.dropna(inplace=True)
        return df


if __name__ == "__main__":
    reader = Read(file_path="NSDUH_2019.parquet")
    df = reader.read()

    df_t0 = df[df['T'] == 0]
    df_t1 = df[df['T'] == 1]

    balance = []
    for c in df_t0.columns:
        balance.append(df_t0[c].value_counts())
        balance.append(df_t1[c].value_counts())

    balance_df = pd.DataFrame(balance)
    balance_df = balance_df / balance_df.sum(axis=1)[:, None]
    balance_df.to_csv(f"Balance_General.csv")
    ...
    # number_of_figs = df.columns.__len__()
    # dim = math.ceil(number_of_figs ** 0.5)
    # fig, ax = plt.subplots(dim, dim, squeeze=False)
    #
    # comb = product(range(dim), range(dim))
    # d = deque(comb)
    #
    # for col in sorted(df.columns):
    #     x, y = d.popleft()
    #     data = df[col].astype(int)
    #     sns.histplot(data, ax=ax[x][y])
    #
    # plt.show()
