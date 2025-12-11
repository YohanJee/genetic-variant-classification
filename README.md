# predictiing genetic variants of uncertain significance

This repository contains all the R codes and data used in the project to predict whether genetic variants have **conflicting clinical significance** using allele frequency features, functional annotations, and predictive scoring features.
The dataset is derived from **ClinVar** and contains 65,188 genetic variants with 46 features.

**Project Overview**

ClinVar provides clinical significance classifications on a five-point scale:

- **Benign**
- **Likely benign**
- **Uncertain/conflicting**
- **Likely pathogenic**
- **Pathogenic**

Our goal is to build machine-learning models that classify whether a variant has **conflicting pathogenicity**.


**Objective**
**Predict whether a variant is “conflicting” (1) or “non-conflicting” (0)**  
—using interpretable biological and computational features.


**Features Used**

### **Allele Frequency Features**
- `AF_ESP`
- `AF_EXAC`
- `AF_TGP`

### **Functional Annotations**
- `Consequence`  
- `IMPACT`  
- `SIFT`  
- `PolyPhen`

### **Predictive Scores**
- `CADD_PHRED`
- `CADD_RAW`
- `LoFtool`
- `BLOSUM62`

## **Handling Class Imbalance**

- **Class weighting:** Increase penalty for misclassifying minority class (conflicting variants)
- **Threshold tuning:** Adjust probability threshold for better sensitivity

## **Models Implemented**

| Model | Versions Included |
|-------|------------------|
| **Logistic Regression** | Baseline, Class Weight, Threshold |
| **Random Forest** | Baseline, Class Weight, Threshold |
| **XGBoost** | Baseline, Class Weight, Threshold |


## **Evaluation Metrics**

- AUC (Area Under ROC Curve)  
- Balanced Accuracy  
- Sensitivity / Recall  
- Specificity  


