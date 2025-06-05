"""
Offline training script:
$ python -m credit_score.models.train
"""
from pathlib import Path
import joblib, pickle

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from credit_score.features.load_data  import load_split
from credit_score.features.cleaning   import fit_cleaning_metadata, clean_and_prepare
from credit_score.features.pipeline   import make_preprocessor


def main():
    # ---------- data ------------------------------------------------------ #
    df_raw   = load_split("train")
    meta     = fit_cleaning_metadata(df_raw)
    df_train = clean_and_prepare(df_raw, meta)

    X_train  = df_train.drop("default", axis=1)
    y_train  = df_train["default"]

    # ---------- preprocessing + model ------------------------------------- #
    preproc  = make_preprocessor()
    X_train_pp = preproc.fit_transform(X_train, y_train)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train_pp, y_train)

    # optional quick AUC on train
    print("train AUC:",
          roc_auc_score(y_train, clf.predict_proba(X_train_pp)[:, 1]).round(4))

    # ---------- persist artefacts ----------------------------------------- #
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(preproc, "artifacts/preprocessor.pkl")
    joblib.dump(clf    , "artifacts/model.pkl")
    pickle.dump(meta   , open("artifacts/cleaning_meta.pkl", "wb"))
    print("Saved artefacts to artifacts/")


if __name__ == "__main__":
    main()
