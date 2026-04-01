# Clinical Decision Support System for DKA Recurrence

## System Overview

The service is designed to predict the risk of diabetic ketoacidosis (DKA) recurrence and provide interpretable clinical insights for physicians.

The architecture includes:
- inference layer (models, explanation, interpretation)  
- API (FastAPI)  
- user interface (Streamlit)  

---

## 1. Inference Layer

### `clinical.py`

#### `ClinicalInterpreter`
A class for clinical interpretation of model predictions.

**Core functionality:**

##### `risk_level(proba)`
- Converts model probability into risk categories:
  - Low risk  
  - Moderate risk  
  - High risk  

---

##### `top_drivers(shap_dict, top_n=5)`
- Identifies key factors influencing the prediction  
- Sorts features by absolute SHAP values  
- Returns top-N features with direction:
  - increases risk  
  - decreases risk  

---

##### `numeric_alerts(deviations)`
- Analyzes numerical deviations  
- Uses z-score  
- Generates alerts:
  - above normal  
  - below normal  

---

##### `categorical_alerts(X_row)`
- Analyzes categorical features  
- Detects rare categories (frequency < 5%)  
- Generates warnings  

---

##### `build_alerts(deviations, X_row)`
- Combines:
  - numerical deviations  
  - rare categories  

---

##### `summary(risk_level)`
- Generates a clinical summary:
  - high risk → requires attention  
  - moderate → monitoring recommended  
  - low → no significant deviations  

---

##### `build_response(...)`
- Constructs the final response:
  - risk level  
  - key drivers  
  - alerts  
  - summary  

---

### `explain.py`

#### `Explainer`
A class for model interpretability.

##### `shap_values(X)`
- Computes SHAP values  

---

##### `feature_importance_patient(X)`
- Returns SHAP values for a specific patient  

---

##### `detect_outliers(X_row, df_reference)`
- Computes z-score for each feature  
- Detects outliers where |z| > 2  
- Returns:
  - value  
  - mean  
  - z-score  

---

### `predictor.py`

#### `DKAPredictor`
A class for ensemble prediction.

##### `predict_proba(X)`
- Averages probabilities across multiple models (folds)  
- Returns final probability  

---

##### `predict(X)`
- Binary prediction:
  - 1 — recurrence  
  - 0 — no recurrence  

---

##### `confidence(proba)`
- Estimates prediction reliability  
- Based on distance from threshold:
  - LOW  
  - MEDIUM  
  - HIGH  

---

### `loader.py`

#### `load_models()`
- Loads ensemble of CatBoost models  
- Uses k-fold approach (5 models)  

---

#### `load_threshold()`
- Loads threshold from JSON  
- Uses median value  

---

## 2. API Layer (FastAPI)

### `app.py`

#### Main logic:
- Loads:
  - dataset  
  - models  
  - threshold  

---

#### Endpoint: `/predict/{medical_record_id}`

##### Performs:
1. Patient existence validation  
2. Feature preparation  
3. Model prediction:
   - probability (`proba`)  
   - class (`prediction`)  
   - confidence  

4. Interpretation:
   - SHAP values  
   - deviations  
   - clinical insights  

5. Dataset membership:
   - train / test / unknown  

---

##### Returns:
- prediction  
- proba  
- confidence  
- dataset  
- clinical:
  - risk_level  
  - drivers  
  - alerts  
  - summary  
- shap  

---

## 3. User Interface (Streamlit)

### `app.py`

#### Core functionality:

##### Input:
- Patient ID (medical record ID)  

---

##### Output:

###### 1. Risk Level
- Risk category  
- Short description  

---

###### 2. Model Metrics
- Prediction (recurrence / no recurrence)  
- Confidence  

---

###### 3. Dataset Info
- train  
- test  
- unknown  

---

###### 4. Key Risk Factors
- Top features (SHAP)  
- Direction of impact  

---

###### 5. Clinical Deviations
- Numerical anomalies  
- Rare categories  

---

###### 6. Visualization
- Risk factor bar chart (Plotly)  
- Top-10 features  

---

###### 7. Model Probability
- `predict_proba` output  
- Numerical risk value  

---

## 4. Data Flow

1. User inputs patient ID  
2. UI sends request to API  
3. API:
   - retrieves data  
   - generates prediction  
   - interprets results  
4. Response is returned to UI  
5. UI visualizes:
   - risk  
   - drivers  
   - deviations  
   - probability  

---

## 5. Key Characteristics

- Interpretability (SHAP + clinical rules)  
- Ensemble models (robustness)  
- Automatic deviation detection  
- Transparent train/test split  
- Clinical decision support  