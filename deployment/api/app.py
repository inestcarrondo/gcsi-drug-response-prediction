from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Optional
import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from pathlib import Path
import sys
sys.path.append('../../ds/model')
from data_science.train import train_single

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- Settings --------
MODEL_PATH = os.getenv("MODEL_PATH", "../../ds/artifacts/hgb_random_smiles/model.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "../../ds/artifacts/hgb_random_smiles/features.json")

app = FastAPI(
    title="gCSI Drug Response Prediction API",
    version="1.0.0",
    description="Predict drug response (GRinf) for drug-cell line combinations using trained ML models"
)

# -------- I/O Schemas --------
class DrugCellSample(BaseModel):
    """Input schema for drug-cell line prediction"""
    norm_drugname: str
    norm_celllinename: str
    primarytissue: Optional[str] = None
    trtduration: Optional[float] = 72.0
    doublingtime: Optional[float] = None
    maxlog10concentration: Optional[float] = None
    gr_aoc: Optional[float] = None
    gr50: Optional[float] = None
    r_square_gr: Optional[float] = None
    h_gr: Optional[float] = None
    n_conc: Optional[float] = None
    log10_conc_step: Optional[float] = None
    smiles: Optional[str] = None

    @field_validator("norm_drugname", "norm_celllinename")
    @classmethod
    def not_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("Drug name and cell line name must not be empty")
        return v.strip()

class PredictRequest(BaseModel):
    samples: List[DrugCellSample]

class PredictResponse(BaseModel):
    predictions: List[float]
    model_info: dict
    processing_time_seconds: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_loaded: bool
    timestamp: str
    model_path: str

# -------- Model Loading --------
_model = None
_features_meta = None

def load_model():
    """Load the trained model and feature metadata, train if missing"""
    global _model, _features_meta
    
    # Resolve paths relative to this file
    base_dir = Path(__file__).parent
    model_path = base_dir / MODEL_PATH
    features_path = base_dir / FEATURES_PATH
    
    try:
        if model_path.exists() and features_path.exists():
            _model = joblib.load(model_path)
            with open(features_path, 'r') as f:
                _features_meta = json.load(f)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.info("Model artifacts not found, training new model...")
            train_model_on_demand()
            
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise

def train_model_on_demand():
    """Train a simple model when artifacts are missing"""
    global _model, _features_meta
    
    try:
        base_dir = Path(__file__).parent
        artifacts_dir = base_dir / "../../ds/artifacts/api_trained"
        
        logger.info("Training Ridge model with basic settings...")
        out_dir, metrics = train_single(
            drug=None,
            out_dir=str(artifacts_dir),
            model_name="ridge",
            split="random",
            include_smiles="no",
            do_cv=False
        )
        
        # Load the newly trained model
        model_path = Path(out_dir) / "model.joblib"
        features_path = Path(out_dir) / "features.json"
        
        _model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            _features_meta = json.load(f)
            
        logger.info(f"Model trained and loaded successfully. R²: {metrics['scores']['r2']:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to train model on demand: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

# -------- Preprocessing --------
def preprocess_samples(samples: List[DrugCellSample]) -> pd.DataFrame:
    """Convert samples to DataFrame with proper column names"""
    data = []
    for sample in samples:
        row = {
            'norm_drugname': sample.norm_drugname,
            'norm_celllinename': sample.norm_celllinename,
            'primarytissue': sample.primarytissue or 'unknown',
            'trtduration': sample.trtduration or 72.0,
            'doublingtime': sample.doublingtime,
            'maxlog10concentration': sample.maxlog10concentration,
            'gr_aoc': sample.gr_aoc,
            'gr50': sample.gr50,
            'r_square_gr': sample.r_square_gr,
            'h_gr': sample.h_gr,
            'n_conc': sample.n_conc,
            'log10_conc_step': sample.log10_conc_step,
            'smiles': sample.smiles
        }
        data.append(row)
    
    return pd.DataFrame(data)

# -------- Endpoints --------
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "gCSI Drug Response Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        features_loaded=_features_meta is not None,
        timestamp=datetime.now().isoformat(),
        model_path=MODEL_PATH
    )

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not _features_meta:
        raise HTTPException(status_code=404, detail="Model metadata not available")
    
    return {
        "model_type": _features_meta.get("model_type", "unknown"),
        "features_count": _features_meta.get("n_features_total", 0),
        "includes_smiles": _features_meta.get("smiles_column") is not None,
        "version": _features_meta.get("version", 1),
        "numeric_features": _features_meta.get("numeric", []),
        "categorical_features": _features_meta.get("categorical", []),
        "smiles_column": _features_meta.get("smiles_column")
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make drug response predictions"""
    if not _model:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check /health endpoint for status."
        )
    
    start_time = datetime.now()
    
    try:
        # Preprocess input
        df = preprocess_samples(request.samples)
        logger.info(f"Processing {len(df)} samples for prediction")
        
        # Make predictions using the loaded pipeline
        predictions = _model.predict(df)
        
        # Convert to list of floats
        pred_list = [float(p) for p in predictions]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictResponse(
            predictions=pred_list,
            model_info={
                "model_type": _features_meta.get("model_type", "unknown") if _features_meta else "unknown",
                "features_used": _features_meta.get("n_features_total", 0) if _features_meta else 0
            },
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)