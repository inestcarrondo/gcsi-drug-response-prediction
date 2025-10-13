#!/bin/bash

# Local deployment script for gCSI Drug Response API

echo "Starting gCSI Drug Response Prediction API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables for local development
export MODEL_PATH="../../artifacts/hgb_random_smiles/model.joblib"
export FEATURES_PATH="../../artifacts/hgb_random_smiles/features.json"

# Start the API server
echo "Starting API server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

uvicorn app:app --host 0.0.0.0 --port 8000 --reload