import pandas as pd
from typing import Dict, Any


def fit_cleaning_metadata(df_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics from df_train only, to apply consistent cleaning rules.
    """
    metadata = {}

    # 1. Most frequent value for residence_type
    metadata["mode_residence_type"] = df_train["residence_type"].mode()[0]

    # 2. Threshold: max acceptable processing_fee as % of loan_amount
    metadata["max_fee_ratio"] = 0.03  # business rule

    # 3. Value typo mapping
    metadata["loan_purpose_fix"] = {"Personaal": "Personal"}

    return metadata


def clean_and_prepare(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Clean and prepare the dataset using metadata from training data only.
    """
    df = df.copy()
    n_before = len(df)
    # 1. Fill missing residence_type with mode from training data
    df["residence_type"] = df["residence_type"].fillna(metadata["mode_residence_type"])

    # 2. Remove outliers in processing_fee using train-based ratio
    mask = (df["processing_fee"] / df["loan_amount"]) <= metadata["max_fee_ratio"]
    df = df[mask].copy()

    # 3. Fix typos in loan_purpose
    df["loan_purpose"] = df["loan_purpose"].replace(metadata["loan_purpose_fix"])
    n_after  = len(df)
    print(f"Outliers removed: {n_before - n_after}")
    return df
