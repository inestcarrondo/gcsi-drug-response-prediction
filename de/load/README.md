# Data Loading Documentation

This folder contains SQL scripts documenting how data is loaded into the database.

## Files

- `load_gcsi_data.sql`: Documents the COPY commands used to load CSV/TSV files into PostgreSQL tables

## Data Loading Process

1. **gcsi_grmetrics**: Loaded from `gCSI_GRmetrics_v1.3.tsv` (tab-delimited)
2. **gcsi_grvalues**: Loaded from `gCSI_GRvalues_v1.3.tsv` (tab-delimited)  
3. **drug_smiles**: Loaded from `drug_smiles.csv` (comma-delimited)

## Usage

The data loading happens automatically when the Docker container starts, but the SQL commands are documented here for reference and manual execution if needed.

```bash
# To manually load data (if needed):
psql -h localhost -U admin -d daredatachallenge -f load_gcsi_data.sql
```