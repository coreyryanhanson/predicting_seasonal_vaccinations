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

def gen_impute_dict(binary, ordinal, continuous):
    impute_dict = OrderedDict()
    for column in binary:
        impute_dict[column] = [np.nan, "most_frequent", None]
    for column in [*ordinal, *continuous]:
        impute_dict[column] = [np.nan, "median", None]
    return impute_dict

def get_imputer_objs(df, impute_dict):
    """Stores a list of fitted imputers to transform the data later."""

    return [SimpleImputer(val[0], val[1], val[2]).fit(df[[key]]) for key, val in impute_dict.items()]

#
def extract_column_names(df, term):
    """Creates a Pandas Index object for column names found via a regex search."""

    matches = [column for column in df.columns if re.search(term, column)]
    return pd.Index(matches)


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

def rename_poverty(df):
    income_dict = {'Below Poverty': 'Below Poverty', '<= $75,000, Above Poverty': 'Above Poverty',
                   '> $75,000': "Above $75k"}
    df['income_poverty'] = df['income_poverty'].map(income_dict)
    return df

def rent_or_own_to_homeowner(df):
    val_dict = {"Own":1, "Rent":0}
    df["rent_or_own"]=df["rent_or_own"].map(val_dict)
    return df.rename(columns={"rent_or_own":"homeowner"})

def redundant_missing_group(df, term):
    redundant_columns = [column for column in extract_column_names(df, "^"+term)]
    redundant_data = [df[column] for column in redundant_columns]
    redundant_sum = sum(redundant_data)
    df[term] = np.where(redundant_sum.values >= 1, 1, 0)
    return df.drop(columns=redundant_columns)

def redundant_missing(df):
    df = redundant_missing_group(df, "missing_opinion")

    df = df.rename(columns={"missing_household_adults": "missing_household"})
    df = df.rename(columns={'missing_doctor_recc_h1n1': 'missing_doctor_recc'})
    df = df.drop(columns="missing_household_children")
    df = df.drop(columns='missing_doctor_recc_seasonal')

    #The following three columns are highly correlated, but do not share any apparent relationship. Since they
    # make up an insiginificant portion of the data, they are all dropped.
    df = df.drop(columns='missing_child_under_6_months')
    df = df.drop(columns='missing_health_worker')
    df = df.drop(columns='missing_chronic_med_condition')

    return df

def initial_cleaning(df):
    df = rent_or_own_to_homeowner(df)
    df = rename_poverty(df)
    return df

def fit_clean_data(df, impute_dict):
    # Fits imputers

    imputes = get_imputer_objs(df, impute_dict)
    return clean_data(df, imputes, impute_dict), imputes

def clean_data(df, imputes, impute_dict):

    df = impute_vals(df, impute_dict, imputes)
    df = categorical_nans(df, "unknown")
    # df["age_group"] = df["age_group"].map(rename_ages)
    df = redundant_missing(df)
    return df
