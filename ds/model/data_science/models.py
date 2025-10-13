import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import joblib
import os

from .config import TARGET, MIN_SAMPLES_PER_DRUG, TEST_SIZE, RANDOM_STATE

def train_model(df, preprocessor, cv_strategy='cell_split', model_type='rf'):
    """
    Train drug response prediction model.
    
    Parameters:
    - df: DataFrame with drug response data
    - preprocessor: sklearn preprocessing pipeline
    - cv_strategy: 'cell_split' or 'drug_split'
    - model_type: 'rf', 'ridge', or 'hgb'
    
    Returns:
    - Trained model and evaluation metrics
    """
    # Filter drugs with sufficient data
    drug_counts = df['norm_drugname'].value_counts()
    valid_drugs = drug_counts[drug_counts >= MIN_SAMPLES_PER_DRUG].index
    df_filtered = df[df['norm_drugname'].isin(valid_drugs)].copy()
    
    print(f"Using {len(valid_drugs)} drugs with ≥{MIN_SAMPLES_PER_DRUG} samples each")
    print(f"Total samples: {len(df_filtered):,}")
    
    # Prepare features
    feature_cols = ['trtduration', 'doublingtime', 'maxlog10concentration',
                   'gr_aoc', 'gr50', 'r_square_gr', 'h_gr', 'n_conc', 'log10_conc_step',
                   'norm_drugname', 'primarytissue']
    
    if 'smiles' in df_filtered.columns:
        feature_cols.append('smiles')
    
    X = df_filtered[feature_cols]
    y = df_filtered[TARGET]
    
    # Split data based on strategy
    if cv_strategy == 'drug_split':
        # Group by drug for cross-drug prediction
        groups = df_filtered['norm_drugname']
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"\\nDrug split - Train drugs: {X_train['norm_drugname'].nunique()}")
        print(f"Drug split - Test drugs: {X_test['norm_drugname'].nunique()}")
        
    else:  # cell_split
        # Random split for within-drug prediction
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"\\nCell line split - random 80/20 split")
    
    # Build model pipeline
    models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'ridge': Ridge(alpha=1.0),
        'hgb': HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    }
    
    model = models.get(model_type, models['rf'])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results = {
        'pipeline': pipeline,
        'r2': r2,
        'spearman': spearman,
        'rmse': rmse,
        'strategy': cv_strategy,
        'model_type': model_type
    }
    
    print(f"\\nResults ({cv_strategy}, {model_type}):")
    print(f"R²: {r2:.3f}")
    print(f"Spearman ρ: {spearman:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    return results

def save_model(pipeline, path):
    """Save trained model pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")

def load_model(path):
    """Load trained model pipeline."""
    return joblib.load(path)