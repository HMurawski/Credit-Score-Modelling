from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .engineering import add_features
from .selectors   import VIFDropper, ColumnSubsetter
from sklearn.impute import SimpleImputer   

SELECTED_FEATURES = [
    "age", "loan_tenure_months", "number_of_open_accounts",
    "credit_utilization_ratio", "loan_to_income",
    "deliquency_ratio", "avg_dpd_per_deliquency",
    "loan_purpose", "residence_type", "loan_type",
]

# num / cat based on feature_engineering
NUM_COLS = [
    "age", "loan_tenure_months", "number_of_open_accounts",
    "credit_utilization_ratio", "loan_to_income",
    "deliquency_ratio", "avg_dpd_per_deliquency",
]
CAT_COLS = ["loan_purpose", "residence_type", "loan_type"]

def make_preprocessor():
    num_pipe = Pipeline([
        ("imp" , SimpleImputer(strategy="median")),
        ("vif"  , VIFDropper(threshold=5.0)),
        ("scal" , MinMaxScaler()),
    ])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    col_t = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])

    return Pipeline([
        ("row_feat", FunctionTransformer(add_features, validate=False)),
        ("subset"  , ColumnSubsetter(SELECTED_FEATURES)),  
        ("columns" , col_t),
    ])
