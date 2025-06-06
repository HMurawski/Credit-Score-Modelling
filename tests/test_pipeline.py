import pandas as pd
import numpy as np
import pytest
from credit_score.features.pipeline import make_preprocessor
from scipy.sparse import issparse

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 40, np.nan],
        "loan_tenure_months": [12, 24, 36],
        "number_of_open_accounts": [3, 5, 4],
        "credit_utilization_ratio": [0.4, 0.6, np.nan],
        "loan_amount": [10000, 20000, 0],
        "income": [5000, 0, 10000],
        "delinquent_months": [2, 0, 3],
        "total_loan_months": [10, 5, 0],
        "total_dpd": [40, 10, 90],
        "loan_purpose": ["Personal", "Personaal", "Home"],
        "residence_type": ["Own", None, "Rent"],
        "loan_type": ["Secured", "Unsecured", "Secured"]
    })

def test_pipeline_runs_on_sample_data(sample_df):
    pipe = make_preprocessor()
    result = pipe.fit_transform(sample_df)

    assert result.shape[0] == sample_df.shape[0]
    assert issparse(result) or isinstance(result, np.ndarray)

def test_pipeline_index_preserved(sample_df):
    pipe = make_preprocessor()
    original_index = sample_df.index
    result = pipe.fit_transform(sample_df)

    assert list(original_index) == list(range(result.shape[0]))

def test_pipeline_no_nan_after_processing(sample_df):
    pipe = make_preprocessor()
    result = pipe.fit_transform(sample_df)

    
    if issparse(result):
        result = result.toarray()

    assert not np.isnan(result).any()
