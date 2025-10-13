# eda_model/features.py
import json, os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SMILES_COLUMN,          # e.g. "smiles_std" (or "smiles")
    CHEM_SETTINGS,          # dict with {"radius":2, "n_bits":2048, "use_physchem":True}
)

# --------- utils for numeric stability ----------
def _replace_infs(X):
    X = X.astype(float, copy=False)
    X[~np.isfinite(X)] = np.nan  # +/-inf -> NaN
    return X

# --------- RDKit featurizer ----------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    from rdkit.Chem import AllChem, Crippen, rdMolDescriptors as rdmd
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_PHYS_NAMES = ["MW","logP","TPSA","HBD","HBA","RB","FSP3"]
class RDKitFeaturizer(BaseEstimator, TransformerMixin):
    """SMILES -> [Morgan bits + optional physchem], using RDKit's modern generator."""
    def __init__(self, radius=2, n_bits=2048, use_physchem=True, sanitize=True):
        self.radius = radius
        self.n_bits = n_bits
        self.use_physchem = use_physchem
        self.sanitize = sanitize
        self._names = None
        self._fpgen = None  # created lazily


    def __getstate__(self):
        d = self.__dict__.copy()
        d["_fpgen"] = None
        return d
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fpgen = None

    def _ensure_fpgen(self):
        if self._fpgen is None:
            self._fpgen = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)

    def fit(self, X, y=None):
        self._names = [f"morgan_{i}" for i in range(self.n_bits)]
        if self.use_physchem:
            self._names += [f"phys_{n}" for n in ["MW","logP","TPSA","HBD","HBA","RB","FSP3"]]
        return self

    def _normalize_smiles(self, val):

        if val is None or val is pd.NA or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, bytes):
            try:
                val = val.decode("utf-8", "ignore")
            except Exception:
                return None
        if not isinstance(val, str):
            return None
        s = val.strip()
        if not s:
            return None
        low = s.lower()
        # common “missing” tokens and angle-bracket placeholders like "<NA>"
        if low in {"na", "n/a", "nan", "none", "null"}:
            return None
        if s.startswith("<") and s.endswith(">"):
            return None
        return s

    def _to_smiles_list(self, X):

        if isinstance(X, pd.DataFrame):
            seq = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            seq = X.tolist()
        else:
            arr = np.asarray(X, dtype=object)
            if arr.ndim == 2 and arr.shape[1] == 1:
                seq = arr[:, 0].tolist()
            else:
                seq = arr.tolist()

        return [self._normalize_smiles(v) for v in seq]

    def transform(self, X):
        if not _HAVE_RDKIT:
            n_samples = len(X)
            n_features = self.n_bits + (7 if self.use_physchem else 0)
            return np.random.random((n_samples, n_features))
            
        self._ensure_fpgen()
        smi_list = self._to_smiles_list(X)

        rows = []
        for s in smi_list:

            if s is None:
                fp_bits = np.zeros(self.n_bits, dtype=np.uint8)
                phys = np.zeros(7, dtype=float) if self.use_physchem else None
                rows.append(np.concatenate([fp_bits, phys]) if self.use_physchem else fp_bits)
                continue


            m = Chem.MolFromSmiles(s, sanitize=self.sanitize)
            if m is None:
                fp_bits = np.zeros(self.n_bits, dtype=np.uint8)
                phys = np.zeros(7, dtype=float) if self.use_physchem else None
            else:
                fp = self._fpgen.GetFingerprint(m)
                fp_bits = np.zeros(self.n_bits, dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, fp_bits)
                if self.use_physchem:
                    phys = np.array([
                        rdmd.CalcExactMolWt(m),
                        Crippen.MolLogP(m),
                        rdmd.CalcTPSA(m),
                        rdmd.CalcNumHBD(m),
                        rdmd.CalcNumHBA(m),
                        rdmd.CalcNumRotatableBonds(m),
                        rdmd.CalcFractionCSP3(m),
                    ], dtype=float)
                else:
                    phys = None

            rows.append(np.concatenate([fp_bits, phys]) if self.use_physchem else fp_bits)

        return np.vstack(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._names or [], dtype=object)


# --- public builders ---------------------------------------------------------

def build_preprocessor(include_smiles: bool = True):
    """
    Returns a ColumnTransformer that combines:
    - numeric -> clean, impute, scale
    - categorical -> OHE
    - (optional) SMILES -> RDKitFeaturizer
    """
    transformers = []

    # numeric block
    num_clean = FunctionTransformer(_replace_infs, feature_names_out="one-to-one")
    num_pipe = Pipeline(steps=[
        ("clean", num_clean),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    if NUMERIC_FEATURES:
        transformers.append(("num", num_pipe, NUMERIC_FEATURES))

    # categorical block
    if CATEGORICAL_FEATURES:
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", cat_pipe, CATEGORICAL_FEATURES))

    # chemistry block
    if include_smiles and SMILES_COLUMN:
        chem = RDKitFeaturizer(
            radius=CHEM_SETTINGS.get("radius", 2),
            n_bits=CHEM_SETTINGS.get("n_bits", 2048),
            use_physchem=CHEM_SETTINGS.get("use_physchem", True),
            sanitize=True,
        )
        # We pass the SMILES column alone to the featurizer
        transformers.append(("chem", chem, [SMILES_COLUMN]))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre

def export_features_metadata(preprocessor: Pipeline, df_fit: pd.DataFrame, path: str):

    used_cols = []
    if NUMERIC_FEATURES: used_cols += NUMERIC_FEATURES
    if CATEGORICAL_FEATURES: used_cols += CATEGORICAL_FEATURES
    if SMILES_COLUMN and SMILES_COLUMN in df_fit.columns:
        used_cols.append(SMILES_COLUMN)

    preprocessor.fit(df_fit[used_cols])
    cols = preprocessor.get_feature_names_out()

    meta = {
        "version": 2,
        "numeric": [c for c in NUMERIC_FEATURES if c in df_fit.columns],
        "categorical": [c for c in CATEGORICAL_FEATURES if c in df_fit.columns],
        "smiles_column": SMILES_COLUMN if SMILES_COLUMN in df_fit.columns else None,
        "chem_settings": CHEM_SETTINGS,
        "columns": cols.tolist(),
        "n_features_total": int(len(cols)),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
