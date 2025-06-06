import pandas as pd
import numpy as np
import pytest
from credit_score.features.engineering import add_features

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "loan_amount": [10000, 20000, 0],
        "income": [5000, 0, 10000],
        "delinquent_months": [2, 0, 3],
        "total_loan_months": [10, 5, 0],
        "total_dpd": [40, 10, 90],
    })

def test_add_features_output_columns(sample_df):
    result = add_features(sample_df)
    for col in ["loan_to_income", "deliquency_ratio", "avg_dpd_per_deliquency"]:
        assert col in result.columns

def test_loan_to_income(sample_df):
    result = add_features(sample_df)
    expected = [2.0, np.nan, 0.0]
    np.testing.assert_array_equal(result["loan_to_income"].values, expected)

def test_deliquency_ratio(sample_df):
    result = add_features(sample_df)
    expected = [20.0, 0.0, np.nan]
    np.testing.assert_allclose(result["deliquency_ratio"].fillna(0), [20.0, 0.0, 0.0])

def test_avg_dpd_per_deliquency(sample_df):
    result = add_features(sample_df)
    expected = [20.0, 0.0, 30.0]
    np.testing.assert_allclose(result["avg_dpd_per_deliquency"], expected)

def test_input_dataframe_is_not_modified(sample_df):
    df_copy = sample_df.copy(deep=True)
    _ = add_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, df_copy)
