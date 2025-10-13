-- Re-create objects cleanly (views first to avoid deps)
DROP VIEW IF EXISTS gcsi_grmetrics_with_smiles CASCADE;
DROP VIEW IF EXISTS gcsi_grvalues_with_smiles CASCADE;

DROP TABLE IF EXISTS drug_smiles CASCADE;
DROP TABLE IF EXISTS gcsi_grmetrics CASCADE;
DROP TABLE IF EXISTS gcsi_grvalues CASCADE;

-- =========================
-- gCSI: Curve SUMMARY METRICS
-- One row per (drug, cell line, experiment)
-- =========================
CREATE TABLE gcsi_grmetrics (
  drugname               TEXT,
  celllinename           TEXT,
  primarytissue          TEXT,
  experimentnumber       TEXT,
  trtduration            INTEGER,
  doublingtime           DOUBLE PRECISION,
  maxlog10concentration  DOUBLE PRECISION,
  gr_aoc                 DOUBLE PRECISION,
  meanviability          DOUBLE PRECISION,
  grmax                  DOUBLE PRECISION,
  emax                   DOUBLE PRECISION,
  grinf                  DOUBLE PRECISION,
  gec50                  DOUBLE PRECISION,
  gr50                   DOUBLE PRECISION,
  r_square_gr            DOUBLE PRECISION,
  pval_gr                DOUBLE PRECISION,
  flat_fit_gr            INTEGER,
  h_gr                   DOUBLE PRECISION,
  n_conc                 INTEGER,
  log10_conc_step        DOUBLE PRECISION,
  norm_celllinename      TEXT,
  norm_drugname          TEXT,
  gr_05um_fit            DOUBLE PRECISION
);

-- =========================
-- gCSI: Curve-Level VALUES
-- One row per (drug, cell line, experiment, concentration)
-- =========================
CREATE TABLE gcsi_grvalues (
  drugname              TEXT,
  celllinename          TEXT,
  primarytissue         TEXT,
  experimentnumber      TEXT,
  trtduration           INTEGER,
  doublingtime          DOUBLE PRECISION,
  log10concentration    DOUBLE PRECISION,
  gr                    DOUBLE PRECISION,
  relative_cell_count   DOUBLE PRECISION,
  activities_mean       DOUBLE PRECISION,
  activities_std        DOUBLE PRECISION,
  norm_celllinename     TEXT,
  norm_drugname         TEXT
);

-- =========================
-- Drug SMILES (populated later by script)
-- =========================
CREATE TABLE drug_smiles (
  norm_drugname TEXT PRIMARY KEY,
  smiles        TEXT
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_grmetrics_drug ON gcsi_grmetrics (norm_drugname);
CREATE INDEX IF NOT EXISTS idx_grmetrics_cell ON gcsi_grmetrics (norm_celllinename);
CREATE INDEX IF NOT EXISTS idx_grvalues_drug  ON gcsi_grvalues  (norm_drugname);
CREATE INDEX IF NOT EXISTS idx_grvalues_cell  ON gcsi_grvalues  (norm_celllinename);

-- Optional convenience views (join metrics/values with SMILES)
CREATE OR REPLACE VIEW gcsi_grmetrics_with_smiles AS
SELECT m.*, s.smiles
FROM gcsi_grmetrics m
LEFT JOIN drug_smiles s USING (norm_drugname);

CREATE OR REPLACE VIEW gcsi_grvalues_with_smiles AS
SELECT v.*, s.smiles
FROM gcsi_grvalues v
LEFT JOIN drug_smiles s USING (norm_drugname);