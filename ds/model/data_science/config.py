import os

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg2://ds_user:ds_user@localhost:5432/daredatachallenge"
)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "../artifacts")
TABLE_METRICS = os.getenv("TABLE_METRICS", "gcsi_grmetrics_with_smiles")

# Default modeling choices
TARGET = os.getenv("TARGET", "grinf")  
NUMERIC_FEATURES = [
    "trtduration", "doublingtime", "maxlog10concentration",
    "gr_aoc", "gr50", "r_square_gr", "h_gr", "n_conc", "log10_conc_step"
]
CATEGORICAL_FEATURES = [
    "norm_drugname", "primarytissue"   
]
MIN_SAMPLES_PER_DRUG = int(os.getenv("MIN_SAMPLES_PER_DRUG", "60"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

SMILES_COLUMN = "smiles"  

# RDKit featurizer settings
CHEM_SETTINGS = {
    "radius": 2,
    "n_bits": 2048,
    "use_physchem": True,
}

