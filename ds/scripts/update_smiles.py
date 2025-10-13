#!/usr/bin/env python3
"""
Populate/refresh SMILES for all drugs present in gCSI tables.

- Queries distinct norm_drugname from gcsi_grmetrics (can switch to UNION with gcsi_grvalues)
- Finds ones missing in drug_smiles (or with NULL smiles)
- Fetches canonical SMILES from PubChem (overrides take precedence if provided)
- Validates with RDKit
- UPSERTs into drug_smiles
- Optionally exports a CSV snapshot for audit

Usage:
  python scripts/update_smiles.py \
      --db-url postgresql+psycopg2://admin:admin@localhost:5432/daredatachallenge \
      --export-csv db/data/drug_smiles.csv \
      --overrides db/data/override_smiles.csv
"""

import argparse
import os
import sys
import time
from typing import Optional, Dict, List, Tuple

import pandas as pd

# --- deps ---
try:
    import pubchempy as pcp
except Exception:
    print("ERROR: pubchempy required. pip install pubchempy", file=sys.stderr)
    raise

try:
    from rdkit import Chem
except Exception:
    print("ERROR: RDKit required. conda install -c conda-forge rdkit", file=sys.stderr)
    raise

try:
    from sqlalchemy import create_engine, text
except Exception:
    print("ERROR: sqlalchemy + psycopg2-binary required. pip install sqlalchemy psycopg2-binary", file=sys.stderr)
    raise


DEFAULT_DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://admin:admin@localhost:5432/daredatachallenge")


def load_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[WARN] overrides file not found: {path}", file=sys.stderr)
        return {}
    df = pd.read_csv(path)
    # accept flexible header case
    cols = {c.lower(): c for c in df.columns}
    need = {"norm_drugname", "smiles"}
    if not need.issubset(set(cols.keys())):
        raise SystemExit("Overrides must have columns: norm_drugname,smiles")
    df = df.rename(columns={cols["norm_drugname"]: "norm_drugname", cols["smiles"]: "smiles"})
    df["norm_drugname"] = df["norm_drugname"].astype(str).str.upper().str.strip()
    df["smiles"] = df["smiles"].astype(str)
    out = {}
    bad = []
    for _, r in df.iterrows():
        s = r["smiles"]
        if Chem.MolFromSmiles(s):
            out[r["norm_drugname"]] = s
        else:
            bad.append(r["norm_drugname"])
    if bad:
        print(f"[WARN] {len(bad)} override SMILES failed RDKit parse (skipped): {bad}", file=sys.stderr)
    return out


def pubchem_smiles(name: str, max_attempts=3, pause=0.5) -> Optional[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            hits = pcp.get_compounds(name, "name")
            if hits:
                smi = hits[0].canonical_smiles
                if smi and Chem.MolFromSmiles(smi) is not None:
                    return smi
            return None
        except Exception:
            if attempt == max_attempts:
                return None
            time.sleep(pause * attempt)
    return None


def fetch_needed_drugs(db_url: str, use_values_union: bool = False) -> pd.DataFrame:
    eng = create_engine(db_url)
    if use_values_union:
        q = text("""
        WITH names AS (
          SELECT DISTINCT UPPER(TRIM(norm_drugname)) AS norm_drugname FROM gcsi_grmetrics WHERE norm_drugname IS NOT NULL
          UNION
          SELECT DISTINCT UPPER(TRIM(norm_drugname)) FROM gcsi_grvalues  WHERE norm_drugname IS NOT NULL
        )
        SELECT n.norm_drugname, s.smiles
        FROM names n
        LEFT JOIN drug_smiles s USING (norm_drugname);
        """)
    else:
        q = text("""
        SELECT d.norm_drugname, s.smiles
        FROM (SELECT DISTINCT UPPER(TRIM(norm_drugname)) AS norm_drugname
              FROM gcsi_grmetrics WHERE norm_drugname IS NOT NULL) d
        LEFT JOIN drug_smiles s USING (norm_drugname);
        """)
    df = pd.read_sql(q, eng)
    df["norm_drugname"] = df["norm_drugname"].astype(str).str.upper().str.strip()
    return df


def upsert_smiles(db_url: str, rows: List[Tuple[str, str]]):
    if not rows:
        return
    eng = create_engine(db_url)
    # Use executemany with ON CONFLICT
    stmt = text("""
    INSERT INTO drug_smiles (norm_drugname, smiles)
    VALUES (:n, :s)
    ON CONFLICT (norm_drugname)
    DO UPDATE SET smiles = EXCLUDED.smiles
    WHERE drug_smiles.smiles IS DISTINCT FROM EXCLUDED.smiles
      AND EXCLUDED.smiles IS NOT NULL;
    """)
    payload = [{"n": n, "s": s} for n, s in rows]
    with eng.begin() as con:
        con.execute(stmt, payload)
        # normalize key defensively
        con.execute(text("UPDATE drug_smiles SET norm_drugname = UPPER(TRIM(norm_drugname));"))
        con.execute(text("ANALYZE drug_smiles;"))


def main():
    ap = argparse.ArgumentParser(description="Populate/refresh drug_smiles from gCSI tables.")
    ap.add_argument("--db-url", default=DEFAULT_DB_URL, help="Postgres URL")
    ap.add_argument("--overrides", help="CSV with columns: norm_drugname,smiles (takes precedence)")
    ap.add_argument("--export-csv", help="Optional path to export a snapshot CSV (e.g., db/data/drug_smiles.csv)")
    ap.add_argument("--union-values", action="store_true", help="Also include drugs seen in gcsi_grvalues")
    args = ap.parse_args()

    overrides = load_overrides(args.overrides)
    df = fetch_needed_drugs(args.db_url, use_values_union=args.union_values)

    missing = df[df["smiles"].isna()]["norm_drugname"].unique().tolist()
    print(f"Found {len(df)} drugs total; {len(missing)} missing SMILES to resolve.")

    resolved: List[Tuple[str, str]] = []
    still_missing: List[str] = []

    for name in missing:
        key = name.upper().strip()
        # overrides first
        smi = overrides.get(key)
        if smi is None:
            smi = pubchem_smiles(key)
        if smi and Chem.MolFromSmiles(smi) is not None:
            resolved.append((key, smi))
        else:
            still_missing.append(key)

    print(f"Resolved {len(resolved)} new SMILES; still missing {len(still_missing)}.")
    if still_missing:
        print("Missing:", still_missing[:25], ("..." if len(still_missing) > 25 else ""))

    # UPSERT new/updated rows
    upsert_smiles(args.db_url, resolved)

    # Optional export (full table)
    if args.export_csv:
        eng = create_engine(args.db_url)
        snap = pd.read_sql(text("SELECT norm_drugname, smiles FROM drug_smiles ORDER BY norm_drugname;"), eng)
        os.makedirs(os.path.dirname(args.export_csv), exist_ok=True)
        snap.to_csv(args.export_csv, index=False)
        print(f"Exported snapshot to {args.export_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
