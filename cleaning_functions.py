

def categorical_nans(df, value):
    """Goes through a dataframe (or selection) and replaces all nans with a single value."""

    for column in df.columns:
        df[column].fillna(value, inplace=True)
    return df
