# eda_model/data_access.py
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from .config import DB_URL, TABLE_METRICS, TARGET, SMILES_COLUMN

def get_engine(db_url: str | None = None):
    return create_engine(db_url or DB_URL)

BASE_QUERY = f"""
SELECT
  m.norm_drugname,
  m.norm_celllinename,
  m.primarytissue,
  m.trtduration,
  m.doublingtime,
  m.maxlog10concentration,
  m.gr_aoc,
  m.gr50,
  m.r_square_gr,
  m.pval_gr,
  m.h_gr,
  m.n_conc,
  m.log10_conc_step,
  m.{SMILES_COLUMN} AS {SMILES_COLUMN},
  {TARGET} AS target
FROM {TABLE_METRICS} AS m
WHERE {TARGET} IS NOT NULL
"""

def load_metrics(engine=None, drug: str | None = None) -> pd.DataFrame:
    eng = engine or get_engine()
    if drug:
        q = text(BASE_QUERY + " AND UPPER(m.norm_drugname) = UPPER(:drug)")
        df = pd.read_sql(q, eng, params={"drug": drug})
    else:
        df = pd.read_sql(text(BASE_QUERY), eng)

    # Turn +/-inf in numerics into NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Normalize textual NA-ish tokens (quotes/whitespace tolerant) in object/string cols
    import re
    MISSING_RE = re.compile(r'^\s*["\']?(?:na|n/?a|none|null|nan)["\']?\s*$', re.IGNORECASE)
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        s = df[c].astype("string")
        mask = s.str.match(MISSING_RE, na=False) | s.isna()
        df.loc[mask, c] = pd.NA

    return df
