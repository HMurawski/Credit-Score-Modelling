import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle
from pathlib import Path
from credit_score.features.cleaning import clean_and_prepare

# ------------------------------------------------------------------ #
# Load artefacts
ROOT = Path(__file__).resolve().parents[2]
ART  = ROOT / "artifacts"                     
PREP = joblib.load(ART / "preprocessor.pkl")
CLF  = joblib.load(ART / "model.pkl")
META = pickle.load(open(ART / "cleaning_meta.pkl", "rb"))
st.sidebar.success(f"Loaded artefacts from → {ART}")

# ------------------------------------------------------------------ #
# Expected cols for pipeline
_EXPECTED = [
    # core 9 fields
    "age", "income", "loan_amount",
    "loan_tenure_months", "avg_dpd_per_deliquency",
    "deliquency_ratio", "credit_utilization_ratio",
    "number_of_open_accounts",
    "residence_type", "loan_purpose", "loan_type",
    #add fee / cashflow columns used in cleaning
    "processing_fee", "gst", "net_disbursement",
    "principal_outstanding", "bank_balance_at_application",
    #other bureau stats occasionally referenced
    "enquiry_count", "total_loan_months", "delinquent_months",
    "total_dpd", "number_of_closed_accounts", "sanction_amount",
]

_NUMERIC_SAFE_ZERO = ["processing_fee", "gst", "net_disbursement"]
def _pad(df: pd.DataFrame) -> pd.DataFrame:      
    """Ensure df has every expected column (filled with NaN)."""
    for col in _EXPECTED:
        if col not in df:
            fill = 0.0 if col in _NUMERIC_SAFE_ZERO else np.nan
            df[col] = fill
    return df[_EXPECTED].copy()

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_and_prepare(_pad(df), META)

def _score(df: pd.DataFrame) -> np.ndarray:
    proba = CLF.predict_proba(PREP.transform(df))[:, 1]
    return np.round(proba, 4)

# ------------------------------------------------------------------ #
st.title("Credit-Score Demo")

#Manual form ---------------------------------------------------- #
with st.form("single"):
    st.subheader("Single Customer")

    c1, c2, c3 = st.columns(3)
    age   = c1.number_input("Age", 18, 100, 30)
    inc   = c2.number_input("Annual Income (PLN)", 0, 1_000_000, 5000, step=100)
    loan  = c3.number_input("Loan amount (PLN)", 0, 1_000_000, 10_000, step=100)

    #live loan-to-income
    lti_ratio = np.nan if inc == 0 else loan / inc
    c3.metric("Loan-to-Income", f"{lti_ratio:.2f}")

    c4, c5, c6 = st.columns(3)
    tenure = c4.number_input("Loan tenure (months)", 0, 240, 12)
    avgdpd = c5.number_input("Avg DPD per delinquency", 0.0, 365.0, 0.0, step=1.0)
    dloans = c6.number_input("Open loan accounts", 0, 50, 3)

    c7, c8, c9 = st.columns(3)
    deli   = c7.number_input("Delinquency ratio (%)", 0.0, 100.0, 0.0, step=0.1)
    cur    = c8.number_input("Credit utilization ratio", 0.0, 5.0, 0.4, step=0.01)
    res    = c9.selectbox("Residence type", ["Owned", "Rented", "Company Provided"])

    c10, c11, c12 = st.columns(3)
    purpose = c10.selectbox("Loan purpose", ["Personal", "Home", "Auto", "Education"])
    ltype   = c11.selectbox("Loan type", ["Secured", "Unsecured"])
    submit  = c12.form_submit_button("Score")

#create dataframe only if button pressed
if submit:
    row = pd.DataFrame(
        [{
            "age": age,
            "income": inc,
            "loan_amount": loan,
            "loan_tenure_months": tenure,
            "avg_dpd_per_deliquency": avgdpd,
            "deliquency_ratio": deli,
            "credit_utilization_ratio": cur,
            "number_of_open_accounts": dloans,
            "residence_type": res,
            "loan_purpose": purpose,
            "loan_type": ltype,
        }]
    )
    prob = float(_score(_clean(row))[0])
    st.success(f"**Default probability:** {prob:.2f}%")

st.divider()

#Batch scoring -------------------------------------------------- #
st.subheader("Batch CSV upload")

sample_path = ROOT / "data" / "raw" / "sample_customers.csv"  
st.download_button(
    "Download sample CSV",
    sample_path.read_bytes() if sample_path.exists() else
    pd.DataFrame(columns=_EXPECTED).to_csv(index=False).encode(),
    file_name="sample_customers.csv",
    mime="text/csv",
)

file = st.file_uploader("Upload CSV with the 9 required columns", type="csv")

if file is not None:
    df_in = pd.read_csv(file)
    probs = _score(_clean(df_in))
    st.dataframe(pd.concat([df_in, pd.Series(probs, name="default_prob")], axis=1))
    st.success(f"Scored {len(probs)} rows.")
