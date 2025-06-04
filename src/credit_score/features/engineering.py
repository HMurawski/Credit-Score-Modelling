import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from existing columns.

    Applies row-wise transformations to enrich the dataset 
    with new features derived from existing ones. It does not rely on 
    external statistics (e.g. means or modes from training data), so it 
    can safely be applied to train, validation, and test sets alike.

    Currently implemented:
    - loan_to_income: ratio of loan amount to declared income (rounded to 2 decimals).
    - deliquency_ratio: percentage of delinquent months relative to total loan months (rounded to 1 decimal).

    Parameters:
    - df: pd.DataFrame
        Input dataframe containing original raw or cleaned features.

    Returns:
    - pd.DataFrame
        A new dataframe copy with added engineered features.
    
    Notes:
    - Future transformations should follow the same row-wise logic, or 
      rely only on metadata derived from the training set (e.g. bin edges, encodings).
    """
    df = df.copy()
    df["loan_to_income"] = round(df["loan_amount"] / df["income"], 2)
    df["deliquency_ratio"] = round(df["delinquent_months"] * 100 / df["total_loan_months"], 1)
    df["avg_dpd_per_deliquency"] = np.where(df["delinquent_months"] != 0, df["total_dpd"] / df["delinquent_months"], 0)
    df["default"] = df["default"].astype(int)
    return df
