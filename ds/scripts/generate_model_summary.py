#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path

def load_model_metrics(artifacts_dir):
    """Load all model metrics from artifacts directory"""
    artifacts_path = Path(artifacts_dir)
    results = []
    
    if not artifacts_path.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        return results
    
    # Find all metrics.json files
    for metrics_file in artifacts_path.rglob("metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract key information
            result = {
                'model': metrics.get('model', 'unknown'),
                'split': metrics.get('split', 'unknown'),
                'used_smiles': metrics.get('used_smiles', False),
                'drug': metrics.get('drug', 'ALL'),
                'r2': metrics.get('scores', {}).get('r2', 0),
                'spearman': metrics.get('scores', {}).get('spearman', 0),
                'rmse': metrics.get('scores', {}).get('rmse', 0),
                'n_train': metrics.get('n_train', 0),
                'n_test': metrics.get('n_test', 0),
                'path': str(metrics_file.parent)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
    
    return results

def generate_markdown_table(results):
    """Generate markdown table from results"""
    if not results:
        return "No model results found."
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Sort by performance (R²) descending
    df = df.sort_values('r2', ascending=False)
    
    # Create markdown table
    table_lines = []
    table_lines.append("| Model | Split Strategy | SMILES | Drug | Test R² | Spearman ρ | RMSE | N Train | N Test | Notes |")
    table_lines.append("|-------|---------------|--------|------|---------|------------|------|---------|--------|-------|")
    
    for _, row in df.iterrows():
        model = row['model'].upper()
        split = row['split'].title()
        smiles = "Yes" if row['used_smiles'] else "No"
        drug = row['drug'] if row['drug'] != 'ALL' else 'Multi-drug'
        r2 = f"{row['r2']:.3f}"
        spearman = f"{row['spearman']:.3f}"
        rmse = f"{row['rmse']:.3f}"
        n_train = f"{row['n_train']:,}"
        n_test = f"{row['n_test']:,}"
        
        # Generate notes based on characteristics
        notes = []
        if row['split'] == 'cellline':
            if row['used_smiles']:
                notes.append("SMILES constant per drug")
            else:
                notes.append("Within-drug prediction")
        elif row['split'] == 'drug':
            if row['used_smiles']:
                notes.append("Cross-drug with structure")
            else:
                notes.append("Cross-drug without structure")
        elif row['split'] == 'random':
            notes.append("Random split baseline")
        
        if row['drug'] != 'ALL':
            notes.append(f"Single drug: {row['drug']}")
            
        notes_str = "; ".join(notes) if notes else ""
        
        table_lines.append(f"| {model} | {split} | {smiles} | {drug} | {r2} | {spearman} | {rmse} | {n_train} | {n_test} | {notes_str} |")
    
    return "\n".join(table_lines)

def main():
    artifacts_dir = "ds/artifacts"
    
    print("Scanning for model artifacts...")
    results = load_model_metrics(artifacts_dir)
    
    if not results:
        print("No model artifacts found. Train some models first:")
        print("cd ds/model && python -m data_science.run_experiments --out ../artifacts --top-n 2 --no-cv")
        return
    
    print(f"Found {len(results)} model results:")
    for result in results:
        print(f"  {result['model']} | {result['split']} | SMILES: {result['used_smiles']} | R²: {result['r2']:.3f}")
    
    print("\n" + "="*80)
    print("MARKDOWN TABLE FOR SOLUTION.MD")
    print("="*80)
    
    table = generate_markdown_table(results)
    print(table)
    
    print("\n" + "="*80)
    print("Copy the table above and replace the Model Performance Summary section in SOLUTION.md")

if __name__ == "__main__":
    main()