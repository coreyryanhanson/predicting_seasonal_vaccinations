import math
import numpy as np
import pandas as pd
import scipy.stats as scs


def calc_z_from_proportions(v1, v2, n1, n2):
    p1, p2, p = v1/n1, v2/n1, (v1+v2)/(n1+n2)
    se1, se2 = p*(1-p)/n1, p*(1-p)/n2
    num = abs(p1 - p2)
    denom = math.sqrt(se1 + se2)
    return num/denom

def calc_chi_squared(df, column, target):
    table = pd.crosstab(df[column], df[target])
    print(table)
    return scs.chi2_contingency(table)

def chi_squared_loop(df, target, alpha):
    failed_cols, pass_cols = [], []
    for column in df.columns:
        chi = calc_chi_squared(df, column, target)
        if chi[1] > alpha:
            failed_cols.append([column, chi[1]])
        else:
            pass_cols.append([column, chi[1]])
    return failed_cols, pass_cols

def z_test_proportions(df, column, target, target_val, alpha):
    failed_vals, passed_vals = [], []
    table = pd.crosstab(df[column], df[target])
    n, n_val = df[target].size, table[target_val].sum()
    for i in np.arange(0, table[target_val].size):
        row = table.iloc[i]
        n1, v1 = row.sum(), row[target_val]
        n2, v2 = n - n1, n_val - v1
        z = calc_z_from_proportions(v1, v2, n1, n2)
        p = scs.norm.cdf(-z) * 2
        if p > alpha:
            failed_vals.append([table.index[i], z, p])
        else:
            passed_vals.append([table.index[i], z, p])
    return failed_vals, passed_vals