import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.impute import SimpleImputer

def categorical_nans(df, value):
    """Goes through a dataframe (or selection) and replaces all nans with a single value."""

    for column in df.columns:
        df[column].fillna(value, inplace=True)
    return df

def get_imputer_objs(df, impute_dict):
    """Stores a list of fitted imputers to transform the data later."""

    return [SimpleImputer(val[0], val[1], val[2]).fit(df[[key]]) for key, val in impute_dict.items()]


def impute_vals(df, impute_dict, imputers):
    """"Loops through imputer object list and creates dummies for missing values. If the dictionary indicates that
    a new categorical value should be added, dummy creation is skipped."""

    for i, column in enumerate(impute_dict.keys()):
        if type(impute_dict[column][2]) != str and impute_dict[column][1] != "constant":
            df["missing_"+ column] = missing_val_dummies(df, column, impute_dict[column][0])
        index = df[column].index
        df[column] = pd.Series(imputers[i].transform(df[[column]]).ravel(), index=index)
    return df


def missing_val_dummies(df, column, bad_data):
    """Creates a new column containing dummy variables indicating what was missing when imputing data."""

    if np.isnan(bad_data):
        return np.where(df[column].isna().values == True, 1, 0)
    else:
        return np.where(df[column].values == bad_data, 1, 0)

def rename_ages(x):
    if x == '65+ Years':
        new = "Over_65"
    else:
        new = re.sub("\s-\s", "_to_", x)
    return new


def fit_clean_data(df, impute_dict):
    # Fits imputers

    imputes = get_imputer_objs(df, impute_dict)
    return clean_data(df, imputes, impute_dict), imputes

def clean_data(df, imputes, impute_dict):

    df = impute_vals(df, impute_dict, imputes)
    df = categorical_nans(df, "unknown")
    df["age_group"] = df["age_group"].map(rename_ages)
    return df
