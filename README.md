# 🧠 Deep Learning for Cognitive Risk Stratification in Atypical Parkinsonism

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

## 📌 Overview

Atypical parkinsonian syndromes (such as PSP, MSA and CBD) often come with cognitive decline that is hard to predict early. This project builds a deep learning pipeline that takes structured biomarker data and classifies individuals as **high** or **low** cognitive risk.

The pipeline covers preprocessing, dimensionality reduction, an ensemble of neural networks, evaluation, and export of per-patient risk scores.

## 🔄 Pipeline

```
Biomarker CSV ─> Imputation + scaling + encoding ─> PCA features ─> 3-model neural network ensemble ─> Risk score + risk level
```

## 🗂️ Data

- **Source:** `[ADD: dataset name / source, and whether it is real or synthetic]`
- **Size:** 5,000 samples, 20 columns
- **Target:** `[ADD: what the target column represents]`
- **Split:** 80% train / 20% test, stratified

The data file is not included in this repository.

## 🧪 Methodology

### 1. Preprocessing
- Numeric features: median imputation and standard scaling
- Categorical features: most-frequent imputation and one-hot encoding
- Built with scikit-learn `Pipeline` and `ColumnTransformer`

### 2. Feature Engineering
- PCA (up to 30 components) to capture combined variance across biomarkers
- PCA components concatenated with the original processed features

### 3. Model
An ensemble of 3 fully connected neural networks, with predictions averaged:

| Layer | Details |
|-------|---------|
| Dense | 512 units, ReLU, Batch Norm, Dropout 0.4 |
| Dense | 256 units, ReLU, Batch Norm, Dropout 0.3 |
| Dense | 128 units, ReLU |
| Output | 1 unit, Sigmoid |

- Optimizer: Adam (learning rate 1e-4)
- Loss: binary cross-entropy
- Balanced class weights
- Early stopping on validation AUC and learning rate reduction on plateau

### 4. Evaluation and Output
- Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix
- Each patient gets a risk score (0 to 1) and a label (High Risk / Low Risk), exported to `Risk_Stratification_Results.csv`
- Models, preprocessor, PCA and metadata are saved for reuse

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.497 |
| ROC-AUC | 0.506 |
| F1 (macro) | 0.46 |

**Confusion matrix (test set, n = 1,000)**

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | 378 | 129 |
| **Actual 1** | 374 | 119 |

### What this means

The current model performs at chance level (AUC ≈ 0.5). It was not able to separate high and low risk patients. Likely reasons:

- The available biomarkers may not carry enough signal for this target on their own
- The target definition may need refinement
- Validation design could be tightened (see Future Work)

## 🔭 Future Work

- Fit preprocessing and PCA on training data only, to avoid data leakage
- Use a separate validation split for early stopping instead of the test set
- Compare against simple baselines (logistic regression, random forest, gradient boosting)
- Feature importance analysis to find which biomarkers carry signal
- Add brain connectome features (structural or functional connectivity) alongside biomarkers
- Cross-validation for more reliable estimates

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Data processing | Pandas, NumPy |
| Preprocessing | scikit-learn (Pipeline, PCA) |
| Deep learning | TensorFlow / Keras |
| Visualization | Matplotlib |

## 👤 Author

**Sushmitha K**
