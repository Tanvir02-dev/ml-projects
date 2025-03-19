import pandas as pd

def preprocess_data(df):
    """
    Handles missing values and encodes categorical variables.
    Fills missing values with the mean and applies one-hot encoding.
    """
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Apply one-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df