from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
import numpy as np
from typing import Dict, Any

@dataclass
class CleaningMeta:
    mode_residence_type: str
    max_fee_ratio: float = 0.03
    loan_purpose_fix: Dict[str, str] | None = None


def fit_cleaning_metadata(df_train: pd.DataFrame) -> CleaningMeta:
    """
    Taking stats from train-split only.
    """
    return CleaningMeta(
        mode_residence_type=df_train["residence_type"].mode()[0],
        loan_purpose_fix={"Personaal": "Personal"},
    )

def clean_and_prepare(df: pd.DataFrame, meta: CleaningMeta) -> pd.DataFrame:
    """
    Clean and prepare the dataset using metadata from training data only.
    """
    df = df.copy()

    #  Fill N/A residence_type
    df["residence_type"] = df["residence_type"].fillna(meta.mode_residence_type)

    # Outlier ratio processing_fee / loan_amount
    denom = df["loan_amount"].replace(0, np.nan)
    mask  = (df["processing_fee"] / denom) <= meta.max_fee_ratio
    outliers = (~mask & denom.notna()).sum()
    if outliers:
        print(f"[cleaning] Removed {outliers} fee outliers")
    df = df[mask | denom.isna()].copy()

    # loan_purpose typos fix
    if meta.loan_purpose_fix:
        df["loan_purpose"] = df["loan_purpose"].replace(meta.loan_purpose_fix)

    return df