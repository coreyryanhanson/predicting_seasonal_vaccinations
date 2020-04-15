import pandas as pd
from scipy.stats import chi2_contingency


def calc_chi_squared(df, column, target):
    table = pd.crosstab(df[column], df[target])
    print(table)
    return chi2_contingency(table)

def chi_squared_loop(df, target, alpha):
    failed_cols = []
    for column in df.columns:
        chi = calc_chi_squared(df, column, target)
        if chi[1] > alpha:
            failed_cols.append([column, chi[1]])
    return failed_cols