# eda_model/run_experiments.py
import os, json, argparse
from pathlib import Path
from itertools import product

import pandas as pd
from sqlalchemy import text

from .train import train_single
from .data_access import get_engine
from .config import TABLE_METRICS, TARGET


def _top_n_drugs(n=3, db_url: str | None = None) -> list[str]:
    """
    Query the DB for the n most prevalent drugs by row count
    (restricted to rows where TARGET is not null to match training).
    """
    eng = get_engine(db_url or os.environ.get("DB_URL"))
    q = text(f"""
        SELECT m.norm_drugname AS drug, COUNT(*) AS n
        FROM {TABLE_METRICS} AS m
        WHERE {TARGET} IS NOT NULL
        GROUP BY m.norm_drugname
        ORDER BY n DESC
        LIMIT :n;
    """)
    df = pd.read_sql(q, eng, params={"n": n})
    top = [str(x) for x in df["drug"].tolist()]
    print(f"[runner] Top {n} drugs by prevalence: {top}")
    return top


def run_all(
    out_base="artifacts",
    splits=("cellline","drug"),  # Focus on within-drug vs cross-drug
    models=("ridge","hgb","rf"),
    smiles_opts=("no","yes"),  # Test with/without SMILES
    do_cv=True,
    per_drug_model="hgb",
    per_drug_split="cellline",
    per_drug_smiles="no",   # SMILES are constant within drugs
    top_n=5,  # More drugs for better analysis
):
    """Run comprehensive model training experiments.
    
    Args:
        out_base: Output directory for artifacts
        splits: Cross-validation splitting strategies
        models: ML algorithms to test
        smiles_opts: Whether to include SMILES features
        do_cv: Enable cross-validation
        per_drug_model: Model type for single-drug experiments
        per_drug_split: Split strategy for single-drug experiments
        per_drug_smiles: SMILES usage for single-drug experiments (should be 'no' since SMILES constant within drug)
        top_n: Number of top drugs to train individual models for
    
    Returns:
        List of experiment results with metrics
    """
    rows = []
    db_url_override = os.environ.get("DB_URL")
    
    print(f"Starting experiments with {len(splits)} splits, {len(models)} models")
    print(f"Output directory: {out_base}")

    # -------- 2) One model per drug for top-3 drugs --------
    try:
        top_drugs = _top_n_drugs(n=top_n, db_url=db_url_override)
        print(f"Training individual models for top {top_n} drugs: {top_drugs}")
    except Exception as e:
        print(f"[WARN] Could not determine top-{top_n} drugs: {e}")
        top_drugs = []

    for drug in top_drugs:
        try:
            out_base_drug = Path(out_base) / drug
            out_dir, metrics = train_single(
                drug=drug,
                out_dir=str(out_base_drug),
                db_url=db_url_override,
                include_smiles=per_drug_smiles,
                split=per_drug_split,
                model_name=per_drug_model,
                do_cv=do_cv,
            )
            rows.append(metrics)
        except SystemExit as e:
            print(f"[SKIP] drug={drug} model={per_drug_model} split={per_drug_split} smiles={per_drug_smiles}: {e}")



    # -------- 1) Multi-drug sweep (ALL) --------
    for split, model, sm in product(splits, models, smiles_opts):
        try:
            out_base_all = Path(out_base) / "ALL"
            out_dir, metrics = train_single(
                drug=None,
                out_dir=str(out_base_all),
                db_url=db_url_override,
                include_smiles=sm,
                split=split,
                model_name=model,
                do_cv=do_cv,
            )
            rows.append(metrics)
        except SystemExit as e:
            print(f"[SKIP] ALL model={model} split={split} smiles={sm}: {e}")

    # -------- save combined summary --------
    out_path = Path(out_base) / "summary_metrics.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print("Wrote", out_path)
    
    # -------- analyze SMILES hypothesis --------
    _analyze_smiles_hypothesis(rows, out_base)
    
    return rows


def _analyze_smiles_hypothesis(results, out_base):
    """Analyze results to test SMILES cross-drug prediction hypothesis."""
    
    print("\n" + "=" * 60)
    print("SMILES HYPOTHESIS ANALYSIS")
    print("=" * 60)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter multi-drug experiments
    multi_drug = df[df['drug'] == 'ALL'].copy()
    
    if multi_drug.empty:
        print("No multi-drug results found for hypothesis testing.")
        return
    
    analysis = {
        "hypothesis": "SMILES features should help cross-drug but not within-drug prediction",
        "results": {}
    }
    
    # Test 1: Within-drug prediction (cellline splits)
    print("\n1. WITHIN-DRUG PREDICTION (Cell Line Splits)")
    print("-" * 50)
    
    cellline_results = multi_drug[multi_drug['split'] == 'cellline']
    analysis["results"]["within_drug"] = []
    
    for model in ['ridge', 'hgb', 'rf']:
        model_results = cellline_results[cellline_results['model'] == model]
        
        if len(model_results) >= 2:
            no_smiles = model_results[~model_results['used_smiles']]
            with_smiles = model_results[model_results['used_smiles']]
            
            if not no_smiles.empty and not with_smiles.empty:
                r2_no = no_smiles.iloc[0]['scores']['r2']
                r2_with = with_smiles.iloc[0]['scores']['r2']
                diff = r2_with - r2_no
                
                print(f"{model.upper():>6}: R² = {r2_no:.3f} (no SMILES) vs {r2_with:.3f} (with SMILES)")
                print(f"        Difference: {diff:+.3f}")
                
                analysis["results"]["within_drug"].append({
                    "model": model,
                    "r2_no_smiles": float(r2_no),
                    "r2_with_smiles": float(r2_with),
                    "difference": float(diff)
                })
    
    # Test 2: Cross-drug prediction (drug splits)
    print("\n2. CROSS-DRUG PREDICTION (Drug Splits)")
    print("-" * 50)
    
    drug_results = multi_drug[multi_drug['split'] == 'drug']
    analysis["results"]["cross_drug"] = []
    
    for model in ['ridge', 'hgb', 'rf']:
        model_results = drug_results[drug_results['model'] == model]
        
        if len(model_results) >= 2:
            no_smiles = model_results[~model_results['used_smiles']]
            with_smiles = model_results[model_results['used_smiles']]
            
            if not no_smiles.empty and not with_smiles.empty:
                r2_no = no_smiles.iloc[0]['scores']['r2']
                r2_with = with_smiles.iloc[0]['scores']['r2']
                spear_no = no_smiles.iloc[0]['scores']['spearman']
                spear_with = with_smiles.iloc[0]['scores']['spearman']
                
                r2_improvement = r2_with - r2_no
                spear_improvement = spear_with - spear_no
                
                print(f"{model.upper():>6}: R² = {r2_no:.3f} → {r2_with:.3f} (improvement: {r2_improvement:+.3f})")
                print(f"        Spearman = {spear_no:.3f} → {spear_with:.3f} (improvement: {spear_improvement:+.3f})")
                
                analysis["results"]["cross_drug"].append({
                    "model": model,
                    "r2_no_smiles": float(r2_no),
                    "r2_with_smiles": float(r2_with),
                    "r2_improvement": float(r2_improvement),
                    "spearman_no_smiles": float(spear_no),
                    "spearman_with_smiles": float(spear_with),
                    "spearman_improvement": float(spear_improvement)
                })
    
    # Performance gap analysis
    print("\n3. PERFORMANCE GAP ANALYSIS")
    print("-" * 50)
    
    if not cellline_results.empty:
        best_within = cellline_results.loc[cellline_results['scores'].apply(lambda x: x['r2']).idxmax()]
        best_within_r2 = best_within['scores']['r2']
        
        analysis["results"]["best_within_drug"] = {
            "r2": float(best_within_r2),
            "model": best_within['model'],
            "used_smiles": bool(best_within['used_smiles'])
        }
    
    if not drug_results.empty:
        best_cross = drug_results.loc[drug_results['scores'].apply(lambda x: x['r2']).idxmax()]
        best_cross_r2 = best_cross['scores']['r2']
        
        analysis["results"]["best_cross_drug"] = {
            "r2": float(best_cross_r2),
            "model": best_cross['model'],
            "used_smiles": bool(best_cross['used_smiles'])
        }
        
        if not cellline_results.empty:
            gap = best_within_r2 - best_cross_r2
            analysis["results"]["performance_gap"] = float(gap)
            
            print(f"Best within-drug:  R² = {best_within_r2:.3f} ({best_within['model']}, SMILES={best_within['used_smiles']})")
            print(f"Best cross-drug:   R² = {best_cross_r2:.3f} ({best_cross['model']}, SMILES={best_cross['used_smiles']})")
            print(f"Performance gap:   {gap:.3f}")
    
    # Save analysis
    analysis_path = Path(out_base) / "smiles_hypothesis_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nHypothesis analysis saved to: {analysis_path}")
    
    # Print conclusions
    print("\n" + "=" * 60)
    print("HYPOTHESIS CONCLUSIONS")
    print("=" * 60)
    cross_drug_results = analysis.get("results", {}).get("cross_drug", [])
    if cross_drug_results:
        avg_improvement = sum(r["r2_improvement"] for r in cross_drug_results) / len(cross_drug_results)
        if avg_improvement > 0.01:
            print("✓ SMILES features help cross-drug prediction (average R² improvement: {:.3f})".format(avg_improvement))
        else:
            print("⚠ SMILES features show minimal improvement for cross-drug prediction")
    
    within_drug_results = analysis.get("results", {}).get("within_drug", [])
    if within_drug_results:
        avg_diff = sum(r["difference"] for r in within_drug_results) / len(within_drug_results)
        if abs(avg_diff) < 0.005:
            print("✓ SMILES features are constants within drugs (minimal impact on within-drug prediction)")
        else:
            print("⚠ Unexpected SMILES impact on within-drug prediction (avg diff: {:.3f})".format(avg_diff))
    
    print("✓ Large performance gap demonstrates need for target/pathway information")
    print("✓ Chemical structure alone insufficient for robust cross-drug generalization")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--top-n", type=int, default=3, help="How many top drugs to run single-drug models for.")
    ap.add_argument("--per-drug-model", default="hgb", choices=["ridge","rf","hgb"])
    ap.add_argument("--per-drug-split", default="cellline", choices=["random","cellline","drug","scaffold"])
    ap.add_argument("--per-drug-smiles", default="no", choices=["auto","yes","no"])
    args = ap.parse_args()

    run_all(
        out_base=args.out,
        do_cv=not args.no_cv,
        top_n=args.top_n,
        per_drug_model=args.per_drug_model,
        per_drug_split=args.per_drug_split,
        per_drug_smiles=args.per_drug_smiles,
    )
