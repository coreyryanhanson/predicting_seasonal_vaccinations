import pandas as pd
from scipy.stats import chi2_contingency


def calc_chi_squared(df, column, target):
    table = pd.crosstab(df[column], df[target])
    print(table)
    return chi2_contingency(table)

