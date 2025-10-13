# gCSI Drug Response Prediction System

A complete end-to-end machine learning system for predicting drug response (GRinf) using the gCSI dataset.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │   DS Module     │    │   API Service   │
│   (PostgreSQL)  │───▶│   (Python)      │───▶│   (FastAPI)     │
│                 │    │                 │    │                 │
│ • Data Storage  │    │ • data_science  │    │ • Predictions   │
│ • SQL Views     │    │   package       │    │ • Health Check  │
│ • Indexing      │    │ • Notebooks     │    │ • Containerized │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- **Python 3.8+** (tested with 3.8-3.11)
- **Docker & Docker Compose** (for database and API deployment)
- **Git** (for cloning the repository)
- **4GB RAM minimum** (8GB recommended)

## Quick Start

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd gcsi_alpha
   
   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Start the system:**
   ```bash
   docker-compose up -d
   ```
   This will:
   - Start PostgreSQL database with gCSI data
   - Launch Adminer web interface at http://localhost:8080
   - Start FastAPI service at http://localhost:8000

3. **Access database (web interface):**
   - URL: http://localhost:8080
   - Server: postgres, Database: daredatachallenge
   - Username: admin, Password: admin

4. **Install DS package:**
   ```bash
   cd ds/model && pip install -e .
   ```

5. **Run analysis:**
   ```bash
   jupyter notebook ds/notebooks/gcsi_analysis_guide.ipynb
   ```

6. **Test API:**
   ```bash
   bash tests/test_api.sh
   ```

## Troubleshooting

**Common issues:**
- **RDKit issues**: Try `conda install -c conda-forge rdkit` if pip fails


## Features

- **Multi-algorithm support**: Ridge, Random Forest, HistGradientBoosting
- **Advanced feature engineering**: RDKit molecular descriptors
- **Cross-validation**: Group-aware splitting (cell line, scaffold)
- **Comprehensive EDA**: 15+ visualizations with statistical analysis
- **Production-ready API**: FastAPI with proper schemas
- **Containerized deployment**: Docker Compose orchestration

## Dataset

- **Source**: gCSI (Genomics of Drug Sensitivity in Cancer)
- **Size**: 16,304 drug-cell line combinations
- **Features**: 44 unique drugs, 549 cell lines, 28 tissue types
- **Target**: GRinf (growth rate at infinite concentration)

## Model Performance

Best performing models achieve:
- **R² > 0.6** on test sets
- **Spearman correlation > 0.7** 
- **Cross-validated performance** with group-aware splits

See `SOLUTION.md` for detailed analysis and results.
