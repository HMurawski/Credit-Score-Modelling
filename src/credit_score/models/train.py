import argparse, json
from pathlib import Path
from datetime import datetime
import joblib, pickle
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.combine  import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

from credit_score.features.load_data  import load_split
from credit_score.features.cleaning   import fit_cleaning_metadata, clean_and_prepare
from credit_score.features.pipeline   import make_preprocessor


# ------------------------------------------------------------------ #
#HELPER – wrap Pipeline so it’s no longer a Pipeline for Imb.
# ------------------------------------------------------------------ #
class _PreprocWrapper(BaseEstimator, TransformerMixin):
    """Sklearn-style transformer that wraps (pipeline-based) preprocessor.

    This allows the "prep" step in the ImbPipeline to NOT be a `Pipeline` object,
    so that imbalanced-learn's pipeline validator doesn't complain.
    """
    def __init__(self, preproc):
        self.preproc = preproc

    def fit(self, X, y=None):
        self.preproc.fit(X, y)
        return self

    def transform(self, X):
        return self.preproc.transform(X)


# ------------------------------------------------------------------ #
#Model factory Optuna
# ------------------------------------------------------------------ #
def _make_model(name: str, **kw):
    if name == "logistic":
        return LogisticRegression(max_iter=2000, n_jobs=-1, **kw)
    if name == "rf":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **kw)
    if name == "xgb":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            **kw,
        )
    raise ValueError(name)


def _optuna_space(trial, name: str):
    if name == "logistic":
        return {
            "C"     : trial.suggest_float("C", 1e-3, 1e3, log=True),
            "tol"   : trial.suggest_float("tol", 1e-6, 1e-2, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["lbfgs", "liblinear", "newton-cg", "saga"]
            ),
        }
    if name == "rf":
        return {
            "n_estimators"      : trial.suggest_int("n_estimators", 300, 800),
            "max_depth"        : trial.suggest_int("max_depth", 4, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features"     : trial.suggest_float("max_features", 0.2, 1.0),
        }
    if name == "xgb":
        return {
            "n_estimators"     : trial.suggest_int("n_estimators", 300, 900),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth"        : trial.suggest_int("max_depth", 3, 10),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0, 10),
        }
    raise ValueError(name)


# ------------------------------------------------------------------ #
#CLI / main flow
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["logistic", "rf", "xgb"],
                        default="logistic")
    parser.add_argument("--optuna",  type=int, default=0,
                        help="#trials (0 = no tuning)")
    parser.add_argument("--cv",      type=int, default=5)
    parser.add_argument("--balance", choices=["on", "off"], default="on")
    parser.add_argument("--out",     default="artifacts")
    args = parser.parse_args()

    #Load & clean data ----------------------------------------------
    raw  = load_split("train")
    meta = fit_cleaning_metadata(raw)
    df   = clean_and_prepare(raw, meta)
    X, y = df.drop("default", axis=1), df["default"]

    preproc_wrap = _PreprocWrapper(make_preprocessor())
    sampler      = None if args.balance == "off" else SMOTETomek(random_state=42)

    #Optuna search (AUC mean CV) ------------------------------------
    best_params = {}
    if args.optuna:

        def objective(trial):
            clf = _make_model(args.model, **_optuna_space(trial, args.model))

            steps = [("prep", preproc_wrap)]
            if sampler is not None:
                steps.append(("smt", sampler))
            steps.append(("clf", clf))
            pipe = ImbPipeline(steps)

            cv = StratifiedKFold(n_splits=args.cv,
                                 shuffle=True, random_state=42)
            proba = cross_val_predict(
                pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            return roc_auc_score(y, proba)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna, show_progress_bar=False)
        best_params = study.best_params
        print(f"[optuna] best AUC={study.best_value:.4f}")
        print("[optuna] best params:", best_params)

    #Final fit on full data -----------------------------------------
    clf_final = _make_model(args.model, **best_params)

    X_pp = preproc_wrap.fit_transform(X, y)
    if sampler is not None:
        X_pp, y = sampler.fit_resample(X_pp, y)

    clf_final.fit(X_pp, y)
    auc_train = roc_auc_score(y, clf_final.predict_proba(X_pp)[:, 1])
    print("Train AUC:", round(auc_train, 4))

    #Persist artefacts ----------------------------------------------
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out) / f"{args.model}_opt{args.optuna}_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(preproc_wrap.preproc, out / "preprocessor.pkl")
    joblib.dump(clf_final,             out / "model.pkl")
    pickle.dump(meta, open(out / "cleaning_meta.pkl", "wb"))
    if best_params:
        json.dump(best_params, open(out / "best_params.json", "w"), indent=2)

    print("Artefacts saved to", out.resolve())


if __name__ == "__main__":
    main()