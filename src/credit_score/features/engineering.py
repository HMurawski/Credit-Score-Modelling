import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from existing columns.

    Applies row-wise transformations to enrich the dataset 
    with new features derived from existing ones. It does not rely on 
    external statistics (e.g. means or modes from training data), so it 
    can safely be applied to train, validation, and test sets alike.

    Parameters:
    - df: pd.DataFrame
        Input dataframe containing original raw or cleaned features.

    Returns:
    - pd.DataFrame
        A new dataframe copy with added engineered features.
    
    """
    df = df.copy()
    df["loan_to_income"] = np.where(df["income"] != 0, df["loan_amount"] / df["income"], np.nan)
    df["deliquency_ratio"] = np.where(df["total_loan_months"] != 0,100 * df["delinquent_months"] / df["total_loan_months"],np.nan,)
    df["avg_dpd_per_deliquency"] = np.where(df["delinquent_months"] > 0,df["total_dpd"] / df["delinquent_months"],0,)
    
    return df
