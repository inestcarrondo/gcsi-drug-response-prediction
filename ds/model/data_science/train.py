# eda_model/train.py
import argparse, json
from pathlib import Path
from typing import Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
from joblib import dump

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import (
    DB_URL, ARTIFACTS_DIR, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_STATE, MIN_SAMPLES_PER_DRUG, SMILES_COLUMN
)
from .data_access import get_engine, load_metrics
from .features import build_preprocessor, export_features_metadata

# Silence RDKit logs early
RDLogger.DisableLog("rdApp.*")


# ------------------------ Helpers ------------------------

def _ensure_min_samples(df: pd.DataFrame, drug: str | None):
    if drug and len(df) < MIN_SAMPLES_PER_DRUG:
        raise SystemExit(
            f"Not enough samples for {drug} (got {len(df)}, need >= {MIN_SAMPLES_PER_DRUG})."
        )

def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns:
        raise SystemExit(
            "Expected alias column 'target' from load_metrics() but it is missing. "
            f"Ensure your SQL selects `{TARGET} AS target`."
        )
    keep = list({*NUMERIC_FEATURES, *CATEGORICAL_FEATURES})
    if SMILES_COLUMN and SMILES_COLUMN in df.columns:
        keep.append(SMILES_COLUMN)
    keep = [c for c in keep if c in df.columns]
    return df[keep + ["target"]].copy()

def _sanitize_smiles(s: pd.Series) -> pd.Series:
    """Return object-dtype series where invalid/missing tokens are None."""
    v = s.astype("object")
    v = v.map(lambda x: x.strip() if isinstance(x, str) else x)

    bad = {"", "none", "nan", "na", "n/a", "null", "<none>", "<nan>", "<na>", "<null>", "<na>", "<NA>"}
    def _norm(x):
        if x is None or x is pd.NA:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        if isinstance(x, str):
            xl = x.lower()
            if xl in bad or (x.startswith("<") and x.endswith(">")):
                return None
            return x
        return None
    return v.map(_norm)

def _safe_mol(sm) -> Chem.Mol | None:
    if sm is None or sm is pd.NA or (isinstance(sm, float) and pd.isna(sm)):
        return None
    if not isinstance(sm, str):
        return None
    s = sm.strip()
    if not s or (s.startswith("<") and s.endswith(">")):
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def _scaffolds_from_smiles(smiles: pd.Series) -> pd.Series:
    s = _sanitize_smiles(smiles)
    out = []
    for sm in s:
        mol = _safe_mol(sm)
        if mol:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            out.append(Chem.MolToSmiles(scaf) if scaf else np.nan)
        else:
            out.append(np.nan)
    return pd.Series(out, index=smiles.index, dtype="object")

def _split_indices(
    df_raw: pd.DataFrame,
    y: pd.Series,
    split: Literal["random","cellline","drug","scaffold"],
    smiles_col: str | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute train/test indices using the raw DataFrame for groups/scaffolds."""
    n = len(df_raw)
    all_idx = np.arange(n)

    if split == "random":
        idx_tr, idx_te = train_test_split(all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return idx_tr, idx_te

    if split == "cellline":
        if "norm_celllinename" not in df_raw.columns:
            raise SystemExit("Group split 'cellline' requires 'norm_celllinename' in the raw dataframe.")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        groups = df_raw["norm_celllinename"]
        idx_tr, idx_te = next(gss.split(all_idx, y, groups=groups))
        return idx_tr, idx_te

    if split == "drug":
        if "norm_drugname" not in df_raw.columns:
            raise SystemExit("Group split 'drug' requires 'norm_drugname' in the raw dataframe.")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        groups = df_raw["norm_drugname"]
        idx_tr, idx_te = next(gss.split(all_idx, y, groups=groups))
        return idx_tr, idx_te

    if split == "scaffold":
        if smiles_col is None or smiles_col not in df_raw.columns:
            raise SystemExit("Scaffold split requires a SMILES column in the raw dataframe.")
        scaffolds = _scaffolds_from_smiles(df_raw[smiles_col]).fillna("NOSCAF")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        idx_tr, idx_te = next(gss.split(all_idx, y, groups=scaffolds))
        return idx_tr, idx_te

    raise ValueError(f"Unknown split: {split}")

def _model_registry(model_name: str):
    if model_name == "ridge":
        est = Ridge(random_state=RANDOM_STATE)
        grid = {"model__alpha": [0.1, 1.0, 10.0]}
    elif model_name == "rf":
        est = RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, max_depth=None
        )
        grid = {"model__max_depth": [None, 10, 20]}
    elif model_name == "hgb":
        est = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        grid = {"model__max_depth": [None, 6, 10], "model__learning_rate": [0.05, 0.1]}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return est, grid

def _score_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        sp = float(spearmanr(y_true, y_pred, nan_policy="omit").correlation)
    except Exception:
        sp = np.nan
    return {"r2": float(r2), "rmse": rmse, "mae": mae, "spearman": sp}


# ------------------------ Training entrypoint ------------------------

def train_single(
    drug: str | None,
    out_dir: str,
    db_url: str | None = None,
    include_smiles: Literal["auto","yes","no"] = "auto",
    split: Literal["random","cellline","drug","scaffold"] = "random",
    model_name: Literal["ridge","rf","hgb"] = "ridge",
    do_cv: bool = True,
):
    # DB connection
    eng = get_engine(db_url or DB_URL)

    # Load & filters
    df_raw = load_metrics(eng, drug=drug)
    if df_raw.empty:
        raise SystemExit("No rows returned from load_metrics(). Check filters/DB.")
    if "target" not in df_raw.columns:
        raise SystemExit(f"Modeling column 'target' not found. Ensure `{TARGET} AS target`.")
    df_raw = df_raw.dropna(subset=["target"]).reset_index(drop=True)
    _ensure_min_samples(df_raw, drug)

    # Decide SMILES usage
    smiles_present = bool(SMILES_COLUMN and (SMILES_COLUMN in df_raw.columns))
    if smiles_present:
        df_raw[SMILES_COLUMN] = _sanitize_smiles(df_raw[SMILES_COLUMN])

    # CRITICAL FIX: Single-drug models should never use SMILES (constant within drug)
    if drug is not None:
        smiles_available = False
        if include_smiles == "yes":
            print(f"WARNING: Forcing include_smiles=no for single-drug model (SMILES constant within {drug})")
    else:
        # Multi-drug models can use SMILES
        if include_smiles == "auto":
            smiles_available = smiles_present and df_raw[SMILES_COLUMN].notna().any()
        elif include_smiles == "yes":
            smiles_available = smiles_present
        else:
            smiles_available = False

    print(
        f"{'Single-drug' if drug else 'Multi-drug'} training (n={len(df_raw)}). "
        f"TARGET='{TARGET}', include_smiles={smiles_available}, split={split}, model={model_name}"
    )
    if drug and include_smiles == "yes":
        print(f"  → SMILES disabled for single-drug {drug} (constant within drug)")

    # Compute split indices
    y_raw = df_raw["target"].astype(float)
    smiles_col_for_split = SMILES_COLUMN if smiles_present else None
    idx_tr, idx_te = _split_indices(df_raw, y_raw, split=split, smiles_col=smiles_col_for_split)

    # Now select feature columns + target for modeling
    df_sel = _select_columns(df_raw)
    X_raw = df_sel.drop(columns=["target"])
    y = df_sel["target"].astype(float)

    # Slice with precomputed indices
    X_tr, X_te = X_raw.iloc[idx_tr], X_raw.iloc[idx_te]
    y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]

    # Build preprocessor with/without SMILES
    pre = build_preprocessor(include_smiles=smiles_available)

    # Estimator + grid
    est, grid = _model_registry(model_name)
    pipe = Pipeline(steps=[("pre", pre), ("model", est)])

    # Cross-validation
    if do_cv:
        if split in ("cellline", "drug", "scaffold"):
            if split == "cellline":
                groups = df_raw["norm_celllinename"].iloc[idx_tr]
            elif split == "drug":
                groups = df_raw["norm_drugname"].iloc[idx_tr]
            else:
                groups = _scaffolds_from_smiles(df_raw[SMILES_COLUMN]).iloc[idx_tr].fillna("NOSCAF")

            n_splits = min(3, max(2, int(pd.Series(groups).nunique())))
            kf = GroupKFold(n_splits=n_splits)
            gscv = GridSearchCV(pipe, grid, cv=kf, scoring="r2", n_jobs=-1, refit=True)
            gscv.fit(X_tr, y_tr, groups=groups)
            pipe = gscv.best_estimator_
            cv_results = {"best_params": gscv.best_params_, "best_score_mean_r2": float(gscv.best_score_)}
        else:
            gscv = GridSearchCV(pipe, grid, cv=3, scoring="r2", n_jobs=-1, refit=True)
            gscv.fit(X_tr, y_tr)
            pipe = gscv.best_estimator_
            cv_results = {"best_params": gscv.best_params_, "best_score_mean_r2": float(gscv.best_score_)}
    else:
        pipe.fit(X_tr, y_tr)
        cv_results = None

    # Evaluate
    y_hat = pipe.predict(X_te)
    scores = _score_dict(y_te.to_numpy(), y_hat)

    # Persist artifacts in a variant-specific folder
    suffix = f"{model_name}_{split}_{'smiles' if smiles_available else 'nosmiles'}"
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir = Path(out_dir) / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(out_dir / "model.joblib")
    dump(pipe, model_path)

    features_path = str(out_dir / "features.json")
    export_features_metadata(pipe.named_steps["pre"], X_tr, features_path)

    metrics = {
        "task": "regression",
        "target_alias": "target",
        "aliased_from": TARGET,
        "scores": scores,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "random_state": int(RANDOM_STATE),
        "drug": drug or "ALL",
        "used_smiles": bool(smiles_available),
        "smiles_column": (SMILES_COLUMN if smiles_available else None),
        "smiles_forced_off": bool(drug is not None and include_smiles == "yes"),
        "split": split,
        "model": model_name,
        "cv": cv_results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", model_path, features_path)
    print("Metrics:", json.dumps(metrics, indent=2))
    return out_dir, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train GR regression model from Postgres.")
    p.add_argument("--drug", type=str, default=None, help="Single drug (e.g., VINBLASTINE).")
    p.add_argument("--out", type=str, default="./artifacts", help="Output directory base.")
    p.add_argument("--db-url", type=str, default=None, help="Override DB URL (optional).")
    p.add_argument("--include-smiles", type=str, choices=["auto","yes","no"], default="auto")
    p.add_argument("--split", type=str, choices=["random","cellline","drug","scaffold"], default="random")
    p.add_argument("--model", type=str, choices=["ridge","rf","hgb"], default="ridge")
    p.add_argument("--no-cv", action="store_true", help="Disable CV/hyperparam search.")
    args = p.parse_args()

    train_single(
        args.drug,
        args.out,
        db_url=args.db_url,
        include_smiles=args.include_smiles,
        split=args.split,
        model_name=args.model,
        do_cv=not args.no_cv,
    )
