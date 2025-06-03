import pandas as pd
from pathlib import Path

def load_split(split_name: str) -> pd.DataFrame:
    """
    Load the merged dataset filtered by train/val/test split.

    Parameters:
    - split_name: "train", "val" or "test"

    Returns:
    - Filtered DataFrame with only rows from selected split.
    """
    root   = Path(__file__).resolve().parents[3]  # project root
    ids    = pd.read_csv(root / "data/split" / f"{split_name}_ids.txt", header=None)[0]

    # load raw pieces
    customers = pd.read_csv(root / "data/raw/customers.csv")
    loans     = pd.read_csv(root / "data/raw/loans.csv")
    bureau    = pd.read_csv(root / "data/raw/bureau_data.csv")

    # Merge and filter
    df = customers.merge(loans, on="cust_id").merge(bureau)
    return df[df["cust_id"].isin(ids)].reset_index(drop=True)
