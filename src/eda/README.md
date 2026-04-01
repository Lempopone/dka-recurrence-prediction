# Preprocessing, Correlation Analysis, Visualization and Transformation Modules

## 1. Preprocessing Functions

### `change_cols_names()`
- Renames Russian column names to English  

---

### `del_cols()`
- Removes:
  - columns with ≥ 60% missing values  
  - duplicate rows  
  - rows with missing target  
  - quasi-constant features  
  - predefined noisy columns  
- Sets medical record ID as index  

---

### `set_dtype()`
- Assigns appropriate data types:
  - `object`  
  - `category`  
  - `float`  
  - `int`  
- Adjusts types if necessary  

---

### `basic_info()`
- Outputs summary table:
  - missing values count  
  - missing percentage  
  - unique values  
  - data types  
  - dataset size (before / after)  

---

## 2. Correlation Functions

### `get_corr_duo()`
- Computes Pearson correlation matrix  
- Extracts top correlated feature pairs  

---

### `get_corr_triad()`
- Searches triads of the form:  
  `a ≈ alpha * b + beta * c`  
- Exhaustive combinations  

---

### `del_corr_duo()`
- Removes highly correlated feature pairs  
- Returns dropped columns  

---

### `del_corr_triad()`
- Processes triads:
  - filters by threshold  
  - computes p-values vs target  
  - removes least significant feature  

---

## 3. Visualization and Anomaly Detection

### `detect_anomaly()`
- Detects anomalies per feature  
- Removes if necessary  

---

### `analyze_shape()`
- Detects multimodal distributions  
- Based on density peaks  

---

### `compute_ecdf(series)`
- Builds empirical CDF  

---

### `analyze_distributions()`

#### Numerical features:
- Histogram  
- Boxplot (by target)  
- Violin plot  
- ECDF  
- Log histogram  
- Metrics:
  - median  
  - mean  
  - variance  
  - IQR  

#### Categorical features:
- Class balance  
- Target distribution  

- Stores feature metadata  

---

## 4. Distribution Transformations

### `decide_transformations()`

Based on:
- skewness  
- multimodality  
- outlier ratio  

Methods:
- `log`  
- `log1p`  
- `yeo_johnson`  
- `reflect_log`  
- `robust`  
- `None`  

---

### `apply_transformations()`
- Applies transformations  
- Returns transformed dataset  

---

## 5. Modeling Pipeline

### `get_models()`
Returns models:
- XGBoost (tree / linear)  
- Logistic Regression  
- LightGBM  
- Random Forest  
- Decision Tree  
- CatBoost  

---

### `get_preprocessor(...)`
Builds preprocessing pipeline:

- `catboost` → passthrough  
- `logreg` → scaling + OHE  
- others → OHE  

---

### `get_optuna_params(...)`
- Hyperparameter generation via Optuna  
- Handles class imbalance  

---

## 6. Threshold Optimization

### `find_best_threshold(...)`

- Searches threshold ∈ [0.01, 0.99]  
- Constraint: recall ≥ target  
- Objective: maximize precision  

Fallback:
- closest recall match  

---

## 7. Nested Cross-Validation

### `train_full_pipeline_nested_cv(...)`

Full training pipeline:

1. Train/test split  
2. Outer CV  
3. Inner CV (Optuna)  
4. Model training  
5. Threshold selection  
6. Model saving  

---

Outputs:
- OOF predictions  
- test predictions  
- metrics  
- best params  
- thresholds  

---

## 8. Model Evaluation

### `summarize_results(...)`

Metrics:
- ROC-AUC  
- standard deviation  

Additional:
- median threshold  
- classification report  

---

## 9. Stacking

### `train_stacking_meta_model(...)`
- Trains meta-model (LogReg)  

---

### `search_best_stacking_combinations(...)`
- Searches model combinations  
- Evaluates via ROC-AUC  

---

## 10. SHAP Analysis

### `run_catboost_shap_ensemble(...)`

- Loads models  
- Computes SHAP  
- Averages results  

Visualization:
- summary plot  
- feature importance  

---

## 11. Key Characteristics

- Nested CV  
- Optuna tuning  
- Class balancing  
- Threshold optimization  
- Ensemble methods  
- SHAP interpretability  