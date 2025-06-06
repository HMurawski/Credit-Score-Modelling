import pandas as pd
import numpy as np
import pytest
from credit_score.features.cleaning import fit_cleaning_metadata, clean_and_prepare, CleaningMeta

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "residence_type": ["Own", None, "Rent", "Own"],
        "loan_amount": [10000, 20000, 0, 15000],
        "processing_fee": [200, 600, 500, 10000],  # last is an outlier
        "loan_purpose": ["Personal", "Personaal", "Home", "Education"]
    })
    
def test_fit_cleaning_metadata(sample_df):
    meta = fit_cleaning_metadata(sample_df)
    assert isinstance(meta, CleaningMeta)
    assert meta.mode_residence_type == "Own"
    assert meta.loan_purpose_fix == {"Personaal": "Personal"}
    
def test_clean_and_prepare_output_shape(sample_df):
    meta = fit_cleaning_metadata(sample_df)
    cleaned = clean_and_prepare(sample_df, meta)
    assert cleaned.shape[0] == 3

def test_fillna_residence_type(sample_df):
    meta = fit_cleaning_metadata(sample_df)
    cleaned = clean_and_prepare(sample_df, meta)
    
    assert cleaned["residence_type"].isna().sum() == 0
    assert meta.mode_residence_type in cleaned["residence_type"].unique()

def test_loan_purpose_fix(sample_df):
    meta = fit_cleaning_metadata(sample_df)
    cleaned = clean_and_prepare(sample_df, meta)
    
    assert "Personaal" not in cleaned["loan_purpose"].values
    assert "Personal" in cleaned["loan_purpose"].values