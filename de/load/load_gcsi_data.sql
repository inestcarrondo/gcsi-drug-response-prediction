-- Load gCSI data from CSV files
-- This documents the data loading process as required by the challenge

-- Load GR metrics data
\copy gcsi_grmetrics FROM '/data/gCSI_GRmetrics_v1.3.tsv' WITH (FORMAT csv, HEADER true, DELIMITER E'\t');

-- Load GR values data  
\copy gcsi_grvalues FROM '/data/gCSI_GRvalues_v1.3.tsv' WITH (FORMAT csv, HEADER true, DELIMITER E'\t');

-- Load drug SMILES data
\copy drug_smiles FROM '/data/drug_smiles.csv' WITH (FORMAT csv, HEADER true);

-- Verify data loading
SELECT 'gcsi_grmetrics' as table_name, COUNT(*) as row_count FROM gcsi_grmetrics
UNION ALL
SELECT 'gcsi_grvalues' as table_name, COUNT(*) as row_count FROM gcsi_grvalues  
UNION ALL
SELECT 'drug_smiles' as table_name, COUNT(*) as row_count FROM drug_smiles;