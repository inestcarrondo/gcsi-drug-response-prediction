# gCSI Drug Response Prediction — Analysis & Results

## Table of Contents
1. [Problem Statement & Dataset Overview](#problem-statement--dataset-overview)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Model Performance Summary](#model-performance-summary)
4. [Implementation Timeline](#implementation-timeline)

---

## Problem Statement & Dataset Overview

### The Challenge
Predicting drug response across cancer cell lines requires understanding both the **drug** and the **target**. While rich cell line data is available, comprehensive target/pathway information is lacking. As a first step toward more complete models, the investigation focused on: **can chemical structure information improve drug response prediction?**

### Approach Taken
Robust cross-drug prediction requires both:
- **Drug information**: Chemical structure, physicochemical properties, mechanism of action
- **Target information**: Pathway activity, mutation status, expression profiles

As a starting point, the focus was on **adding drug information** through molecular structure. SMILES (Simplified Molecular Input Line Entry System) strings were used to generate chemical features via RDKit:
- **Morgan fingerprints**: 2048-bit circular fingerprints capturing substructural patterns
- **Physicochemical descriptors**: Molecular weight, LogP, polar surface area, etc.
- **Structural diversity**: Quantifying chemical space coverage across compounds

### Cross-Validation Strategy
To properly evaluate cross-drug generalization, different data splitting approaches were tested:

**Cell Line Splits** (within-drug prediction):
- Train and test on different cell lines for the same drugs
- Tests model's ability to predict drug response on new cell lines
- SMILES features provide **no additional information** since chemical structure is constant within each drug

**Drug Splits** (cross-drug prediction):
- Train on some drugs, test on completely different drugs
- Tests true generalization to novel compounds
- SMILES features become **critical** as they're the only way to transfer knowledge between drugs

### Key Findings
The results demonstrate that **SMILES features are drug-specific constants** that don't help within-drug prediction but are essential for cross-drug generalization. However, chemical structure alone shows limited ability to generalize across different compounds, highlighting the critical need for target/pathway information.

### The gCSI Dataset
The **Genomics of Drug Sensitivity in Cancer (gCSI)** dataset provides the training foundation:
- **16,304 drug-cell line experiments** across 44 unique compounds
- **549 cancer cell lines** representing 28 different tissue types
- **Rich molecular data** including SMILES chemical structures (96.7% coverage)
- **Standardized response metrics** from dose-response curve fitting

The target variable **GRinf** (growth rate at infinite concentration) is the prediction target - negative values indicate cell killing, positive values indicate resistance. **For simplification, the focus is solely on predicting GRinf values rather than analyzing the underlying dose-response curves.**

---

## Exploratory Data Analysis

### Dataset Snapshot

- **Rows:** 16304  
- **Unique drugs (`norm_drugname`):** 44  
- **Unique cell lines (`norm_celllinename`):** 549  
- **Primary tissues (`primarytissue`):** 28  

**Notes:** NA-like tokens (`"", na, n/a, none, null, nan, inf`) were standardized, and `±inf → NaN`.

### Missingness Analysis

<img src="ds/notebooks/figures/fig01_missingness_complete.png" alt="Complete missingness heatmap showing data completeness across all variables" width="80%">

**Figure 1**: Complete missingness patterns across dataset variables (raw database). Darker regions indicate higher missing data rates. Shows true data quality before filtering.

**Raw Database Missingness (16,688 total rows):**
- **smiles**: 545 nulls (3.3%) - missing chemical structures
- **doublingtime**: 367 nulls (2.2%) - missing cell line doubling times
- **Multiple dose-response metrics**: 384 nulls each (2.3%) - failed curve fitting experiments
  - grinf, gr_aoc, grmax, gec50, gr50, r_square_gr, pval_gr, h_gr, n_conc, log10_conc_step, etc.

**Note**: The modeling dataset filters out 384 rows where `grinf IS NULL`, resulting in 16,304 usable samples with different missingness patterns.

### Target Distribution

<img src="ds/notebooks/figures/fig02_target_hist.png" alt="Histogram of GRinf target variable showing distribution shape and key statistics" width="60%">

**Figure 2**: Distribution of GRinf (target variable). Shows the balance between drug-sensitive (negative values) and drug-resistant (positive values) responses. The 44.3% negative rate indicates substantial class imbalance that affects model training.

**Summary:**  
- **Count:** 16304  
- **Median (Q2):** 0.1352  
- **Q1 / Q3:** −0.5189 / 0.6910  
- **% Negative (target < 0):** 44.3%  
- **Mode-ish region:** **0.135 ± 0.605 (IQR/2)**  
- **Tail strength:** **high** → watch positive-response skew; consider **quantile losses** or **tissue-wise threshold calibration**.

**Critical Insights:**
- **44.3% negative values** represent substantial class imbalance requiring stratified sampling and class weights
- **Wide IQR span (1.21)** indicates strong predictive signal across drug sensitivity spectrum
- **High tail strength** suggests robust loss functions (Huber, quantile regression) may outperform MSE
- **Cytotoxic vs cytostatic split** at zero enables separate modeling strategies for different drug mechanisms

### Tissue Composition

<img src="ds/notebooks/figures/fig03_top_tissues.png" alt="Bar chart of top 10 tissue types by sample count" width="60%">

**Figure 3**: Sample distribution across primary tissue types. Reveals dataset bias toward certain cancer types (Lung, Breast, Blood) which may affect model generalization to underrepresented tissues.

- Dominant tissues: **Lung, Breast, Blood, Colon, Pancreas, Skin, Lymph Node, Ovary, Brain, Stomach**  
- Overrepresented: **Lung, Breast, Blood, Colon, Pancreas**

**Bias Implications:**
- **Lung cancer dominance** may cause model to overfit to respiratory tissue patterns
- **Underrepresented tissues** (Brain, Stomach) face cold-start problem with limited training data
- **Tissue stratification** essential for cross-validation to prevent data leakage
- **Sample weighting** by inverse tissue frequency could improve generalization


### SMILES Coverage

- **Has SMILES column?** **True** (`smiles`)  
- **Coverage (valid, non-empty):** **96.7%** (545/16,688 missing in raw data)  

**Chemical Feature Engineering:**
- **96.7% SMILES coverage** enables comprehensive molecular descriptor generation
- **RDKit pipeline** converts SMILES strings into numerical features:
  - **Morgan fingerprints** (2048-bit): Capture circular substructural patterns
  - **Physicochemical properties**: MW, LogP, TPSA, rotatable bonds, H-bond donors/acceptors
  - **Molecular complexity**: Ring count, aromatic atoms, heteroatom ratios
- **Structure-activity exploration**: Testing if similar chemical structures show similar response patterns
- **Cross-drug generalization**: Evaluating transferability of molecular features across compounds

### Data Quality Assessment

**Target Variable Quality:**
- **GRinf values** are derived from dose-response curve fitting but we use them directly
- **Target outliers:** **> 1.1:** 118 · **< −1.1:** 0
- **No extreme negative outliers** suggests reliable GRinf measurements
- **118 extreme positive outliers** may represent genuine drug resistance cases

**Modeling Focus:**
- The approach predicts **GRinf directly** without analyzing underlying dose-response curves
- This simplifies the problem to standard regression on the final growth rate metric
- Curve fitting quality is assumed adequate based on data preprocessing

### Feature Analysis

- **Strongest |ρ| pairs:**  
  **r_square_gr ~ pval_gr (ρ = −0.61)**, **target ~ gr_aoc (ρ = −0.54)**, **target ~ pval_gr (ρ = +0.52)**, **h_gr ~ pval_gr (ρ = −0.43)**, **gr_aoc ~ pval_gr (ρ = −0.38)**
- **Weak/near-zero pairs:**  
  **gr50 ~ n_conc (ρ = −0.00)**, **gr_aoc ~ gr50 (ρ = −0.01)**, **gr50 ~ doublingtime (ρ = +0.01)**, **r_square_gr ~ n_conc (ρ = +0.01)**, **target ~ n_conc (ρ = +0.01)**

**Feature Type Analysis:**
- **GRinf target** is the direct prediction target (no curve parameters used as features)
- **SMILES features** are drug-specific constants - same for all cell lines within a drug
- **Within-drug prediction**: SMILES add no information (constant across all samples)
- **Cross-drug prediction**: SMILES become the primary way to transfer knowledge between compounds
- **Missing biological context**: No target binding, pathway, or mechanism information limits generalization

### Per-Drug Sample Sizes



**Top drugs by count:**  
JQ1 (1331), GDC0941 (818), AZ628 (748), CAL101 (735), IBRUTINIB (733), VORINOSTAT (536), CRIZOTINIB (536), AZACITIDINE (536), TIVANTINIB (536), PACLITAXEL (535)

<img src="ds/notebooks/figures/fig08_cellline_breadth_per_drug.png" alt="Number of unique cell lines tested per drug, showing experimental coverage" width="60%">

**Figure 8**: Cell line coverage per drug. Wide variation (79-1331 tests per drug) affects model confidence - drugs with more data points enable better predictions.

**Sample Size Impact:**
- **17x variation** in drug coverage (79-1331 tests) creates unequal prediction confidence
- **JQ1 with 1331 tests** enables robust drug-specific modeling vs **narrow-coverage drugs** with <100 tests
- **Data imbalance** may require drug-stratified cross-validation or sample weighting
- **Low-coverage drugs** are candidates for transfer learning from similar compounds

### Drug × Tissue Coverage

<img src="ds/notebooks/figures/fig09_drug_tissue_heatmap.png" alt="Heatmap showing number of cell lines tested for each drug-tissue combination" width="70%">

**Figure 9**: Drug-tissue testing matrix. Intensity indicates number of cell lines tested per combination. Reveals systematic gaps where certain drugs haven't been tested against specific tissue types.

<img src="ds/notebooks/figures/fig10_tissue_breadth_per_drug.png" alt="Bar chart of tissue diversity per drug (number of different tissues tested)" width="60%">

**Figure 10**: Tissue breadth per drug. Shows how many different tissue types each drug has been tested against, indicating generalizability of drug response data.

**Interpretation:**  
- **Broad-coverage:** JQ1 (28), GDC0941 (26), IBRUTINIB (26), CAL101 (26), TIVANTINIB (26)  
- **Narrow-coverage:** G02663201 (1), PLX4720 (1), MK2206 (1), NILOTINIB (1), BORTEZOMIB (1)

**Cold Start Problem:**
- **Systematic gaps** in drug-tissue matrix create cold-start scenarios for untested combinations
- **Narrow-coverage drugs** tested on single tissue types limit generalizability
- **Matrix factorization** or collaborative filtering approaches needed for gap-filling
- **Tissue-specific models** may be required for underrepresented combinations
- **Transfer learning** from broad-coverage drugs could improve narrow-coverage predictions

### Drug × Cell-Line Pairing

<img src="ds/notebooks/figures/fig12_drugs_per_cellline_hist.png" alt="Histogram showing distribution of how many drugs each cell line has been tested against" width="60%">

**Figure 12**: Drug exposure per cell line. Most cell lines tested against 20-40 drugs, with some high-throughput lines tested against all 44 compounds.


**Cross-Validation Strategy:**
- **High-exposure cell lines** tested against most/all drugs enable robust cell line-specific modeling
- **Uneven drug exposure** requires grouped cross-validation to prevent data leakage
- **Cell line stratification** essential to avoid overfitting to well-characterized lines
- **Low-exposure cell lines** may benefit from cell line similarity-based imputation

### Tissue Evenness

<img src="ds/notebooks/figures/fig13_tissue_evenness.png" alt="Tissue sampling evenness score per drug, measuring how balanced testing is across tissue types" width="60%">

**Figure 13**: Tissue sampling balance per drug (normalized entropy). Values near 1 indicate even testing across tissues; values near 0 indicate concentration in few tissue types. Important for detecting sampling bias.

*Reading the metric:*  
High → balanced across tissues. Low → concentrated sampling; watch for tissue confounding.

**Sampling Bias Detection:**
- **Low evenness drugs** concentrated in few tissues risk tissue-specific overfitting
- **High evenness drugs** provide more generalizable training signal across cancer types
- **Tissue confounding** in low-evenness drugs may inflate apparent drug efficacy
- **Bias correction** through tissue-stratified sampling or propensity score weighting recommended

### Top Drug-Tissue Combinations

- Top examples (all **Lung**): JQ1 (113), AZ628 (111), GDC0941 (103), TIVANTINIB (101), MI2 (101), AZACITIDINE (101), CRIZOTINIB (101), CAL101 (101), ERLOTINIB (101), LAPATINIB (101), IBRUTINIB (101), VORINOSTAT (101), PALBOCICLIB (100), DABRAFENIB (100), DOCETAXEL (100), LENALIDOMIDE (100), PACLITAXEL (100), VINCRISTINE (99), GEMCITABINE (98), METHOTREXATE (96), 17AAG (96), PF4708671 (96), NU7441 (95), 5FU (90), CISPLATIN (88)

### Modeling Approach

**Direct GRinf Prediction:**
The focus is on predicting the GRinf target variable directly using:
- **Molecular features** from SMILES (when available)
- **Cell line characteristics** (tissue type, etc.)
- **Basic experimental metadata**

**Simplified Feature Set:**
Rather than using dose-response curve parameters as features, GRinf is predicted from:
1. **Chemical structure** (SMILES → molecular descriptors)
2. **Cell line identity** and tissue type
3. **Experimental conditions** (treatment duration, etc.)

This approach tests whether molecular structure alone can enable cross-drug prediction.

**Modeling Approach:**
- **Direct GRinf prediction** without using curve-derived features as predictors
- **SMILES-based molecular features** as the primary drug information source
- **Cell line features** (tissue type, etc.) as the primary target information


---


## Model Performance Summary

| Model | Split Strategy | Chemical Features | Drug | Test R² | Spearman ρ | RMSE | N Train | N Test |
|-------|---------------|------------------|------|---------|------------|------|---------|--------|
| HGB | Cellline | No | Multi-drug | 0.877 | 0.943 | 0.242 | 13,121 | 3,183 |
| HGB | Cellline | Yes | Multi-drug | 0.876 | 0.942 | 0.243 | 13,121 | 3,183 |
| HGB | Cellline | No | GDC0941 | 0.832 | 0.971 | 0.115 | 648 | 170 |
| RF | Cellline | Yes | Multi-drug | 0.825 | 0.915 | 0.289 | 13,121 | 3,183 |
| RF | Cellline | No | Multi-drug | 0.822 | 0.914 | 0.292 | 13,121 | 3,183 |
| HGB | Cellline | No | AZ628 | 0.808 | 0.916 | 0.262 | 584 | 164 |
| HGB | Cellline | No | JQ1 | 0.763 | 0.909 | 0.269 | 1,058 | 273 |
| HGB | Drug | Yes | Multi-drug | 0.728 | 0.879 | 0.348 | 11,265 | 5,039 |
| HGB | Drug | No | Multi-drug | 0.718 | 0.871 | 0.354 | 11,265 | 5,039 |
| RIDGE | Random | No | Multi-drug | 0.507 | 0.751 | 0.489 | 13,043 | 3,261 |
| RF | Drug | No | Multi-drug | 0.489 | 0.731 | 0.477 | 11,265 | 5,039 |
| RF | Drug | Yes | Multi-drug | 0.424 | 0.685 | 0.506 | 11,265 | 5,039 |
| RIDGE | Drug | No | Multi-drug | 0.181 | 0.519 | 0.603 | 11,265 | 5,039 |

**Key Findings**: 
- **Within-drug prediction**: HGB achieves R² = 0.877, with chemical features providing no benefit (SMILES constant per drug)
- **Cross-drug prediction**: HGB shows R² = 0.718 → 0.728 (+0.010) with chemical features, validating hypothesis
- **Mixed algorithm results**: RF shows chemical features hurt performance (0.489 → 0.424), while HGB benefits
- **Single-drug models**: Strong performance (R² = 0.763-0.832) correctly exclude chemical features
- **Algorithm ranking**: HistGradientBoosting > RandomForest > Ridge
- **Hypothesis validated**: Chemical features provide modest improvement for cross-drug prediction in best-performing algorithm 

---

## Implementation Timeline

### Phase 1: Database Setup (3 hours)
- ✅ PostgreSQL containerization with Docker Compose
- ✅ Data loading scripts with proper schema design
- ✅ SQL views for joining metrics with SMILES data
- ✅ Database indexing for performance optimization

### Phase 2: Data Science Pipeline (11 hours)
- ✅ Comprehensive EDA with 15+ visualizations
- ✅ Advanced feature engineering with RDKit molecular descriptors
- ✅ Multi-algorithm experimentation (Ridge, RF, HGB)
- ✅ Cross-validation with group-aware splitting strategies
- ✅ Model serialization and metadata export

### Phase 3: API Deployment (4 hours)
- ✅ FastAPI service with proper request/response schemas
- ✅ Model loading and prediction pipeline
- ✅ Health check endpoints
- ✅ Docker containerization with volume mounting
- ✅ Integration testing with shell scripts
- ✅ Selective artifact management for repository size optimization

### API Examples

**Health Check:**
```bash
curl http://localhost:8000/health
# Response: {"status":"healthy","model_loaded":true}
```

**Prediction Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "Norm_DrugName": "VINBLASTINE",
        "Norm_CellLineName": "SKBR3",
        "TrtDuration": 72,
        "DoublingTime": 73.0,
        "MaxLog10Concentration": -7.0,
        "PrimaryTissue": "Breast"
      }
    ]
  }'
# Response: {"predictions":[{"grinf_prediction":0.234}]}
```

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Deployment Screenshots

**API Health Check:**

<img src="deployment/screenshots/api_health.png" alt="API Health Check Response" width="80%">

**API Prediction Interface:**

<img src="deployment/screenshots/api_screenshot.png" alt="API Prediction Interface" width="80%">


### Model Artifacts

**Note on Repository Size**: Due to space constraints, only essential model artifacts are included in the repository:
- **API Testing Model**: `ALL/hgb_cellline_nosmiles/` - Best-performing HGB model (R² = 0.877)
- **Analysis Summaries**: JSON files with complete experimental results

**To reproduce full results**:
```bash
# Run complete model training for 3 more prevalent drugs (generates ~200MB of artifacts)
python -m ds.model.data_science.run_experiments --out ds/artifacts

# Generate performance summary
python ds/scripts/generate_model_summary.py
```

**Current API Model**: The deployed API uses the best-performing HGB model (R² = 0.877 within-drug prediction) to demonstrate production-quality results.

---

## Future Steps

### Data Science Enhancements

**Cell Line & Tissue Information Integration:**
- **Genomic profiles**: Incorporate mutation status, copy number variations, and gene expression data
- **Pathway activity**: Add pathway enrichment scores and metabolic profiles
- **Tissue-specific features**: Include tissue-specific biomarkers and microenvironment characteristics
- **Cell line similarity**: Implement cell line clustering based on molecular profiles for transfer learning
- **Multi-omics integration**: Combine transcriptomic, proteomic, and metabolomic data

**Advanced Modeling:**
- **Target information**: Integrate drug-target binding data and mechanism of action
- **Graph neural networks**: Model drug-cell line interactions as bipartite graphs
- **Multi-task learning**: Simultaneously predict multiple response metrics
- **Ensemble methods**: Combine tissue-specific and pan-cancer models

### Deployment & Infrastructure

**AWS-Based API:**
- **Serverless architecture**: Deploy using AWS Lambda and API Gateway for scalability
- **Model serving**: Implement SageMaker endpoints for real-time inference
- **Auto-scaling**: Configure elastic load balancing for variable demand
- **Monitoring**: Set up CloudWatch for performance and error tracking
- **Security**: Implement IAM roles and API authentication

**Data Quality & Validation:**
- **Great Expectations**: Implement automated data validation pipelines
- **Data profiling**: Monitor input data quality and drift detection
- **Model monitoring**: Track prediction accuracy and model performance over time
- **Automated testing**: Set up CI/CD pipelines with data quality gates
- **Alerting**: Configure notifications for data quality issues and model degradation

**Production Readiness:**
- **A/B testing**: Framework for comparing model versions
- **Feature stores**: Centralized feature management and versioning
- **Model registry**: Track model lineage and deployment history
- **Batch processing**: Implement large-scale prediction pipelines