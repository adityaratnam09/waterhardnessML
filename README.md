# waterhardnessML

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21315615.svg)](https://doi.org/10.5281/zenodo.21315615)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-GPLv3-blue.svg)

Machine learning framework for interpretable river water hardness classification from routine physicochemical measurements.

---

## Overview

> **An Interpretable Machine Learning Framework for River Water Hardness Classification**

The project demonstrates that inexpensive field measurements can accurately classify river water hardness without laboratory EDTA titration.

Using measurements collected from the Matanza-Riachuelo River Basin (Argentina), four machine learning models are evaluated:

- Random Forest
- Support Vector Machine
- Logistic Regression
- K-Nearest Neighbours

The Random Forest classifier achieves

- **95.45% Test Accuracy**
- **AUC = 0.995**
- **89.84 ± 4.80% Five-fold Cross Validation Accuracy**

A detailed paper describing the framework is available on Zenodo:

https://doi.org/10.5281/zenodo.21315615

---

## Features

The single Python source file reproduces the complete analysis presented in the paper.

It performs

- data preprocessing
- missing value imputation
- train/test split
- feature scaling
- classifier training
- cross-validation
- ROC analysis
- learning curves
- confusion matrix
- PCA visualisation
- feature importance analysis
- permutation importance
- SHAP explanations
- feature ablation
- Pearson correlation analysis
- minimal sensor suite evaluation

Every figure appearing in the paper is generated automatically.

---

## Repository Structure

```
waterhardnessML/
│
├── waterhardnessML.py
├── README.md
└── LICENSE
```

---

## Requirements

Python 3.10+

Required packages

```
numpy
pandas
matplotlib
scikit-learn
shap
scipy
```

Install using

```bash
pip install numpy pandas matplotlib scikit-learn shap scipy
```

---

## Running the Code

Simply execute

```bash
python waterhardnessML.py
```

This performs the complete workflow from preprocessing through model interpretation and saves all publication figures.

---

## Generated Figures

Running the script generates the figures presented in the accompanying paper, including

- Figure 1 — Class Distribution
- Figure 2 — Model Accuracy Comparison
- Figure 3 — ROC Curves
- Figure 4 — Cross Validation Results
- Figure 5 — Learning Curve
- Figure 6 — Confusion Matrix
- Figure 7 — PCA Decision Boundaries
- Figure 8 — Gini Feature Importance
- Figure 9 — Permutation Importance
- Figure 10 — SHAP Summary Plot
- Figure 11 — Correlation Matrix
- Figure 12 — Minimal Feature Set ROC Comparison

---

## Results

| Model | Accuracy |
|--------|---------:|
| Random Forest | **95.45%** |
| SVM | 79.55% |
| Logistic Regression | 72.73% |
| KNN | 70.45% |
| EC Threshold Baseline | 75.00% |
| Majority Baseline | 71.40% |

Random Forest substantially outperforms both conventional field methods and competing machine learning classifiers.

---

## Minimal Sensor Suite

One of the principal findings is that only three routine measurements are required to match the performance of the complete eleven-feature model:

- Electrical Conductivity (EC)
- Sample Temperature
- Total Suspended Solids (TSS)

---

## Dataset

The implementation uses the publicly available River Water Parameters dataset collected from the Matanza-Riachuelo River Basin, Argentina.

Original dataset:

https://www.kaggle.com/datasets/natanaelferran/river-water-parameters

---

## License

This project is released under the GNU General Public License v3.0.

---

## Citation

If you use this software in research, please cite both the software and the accompanying journal publication.

**Software (GitHub):**

```text
Ratnam, A. R. (2026). *waterhardnessML* (Version 1.0.0) [Computer software]. GitHub. https://github.com/adityaratnam09/waterhardnessML
```

**Journal publication (Zenodo):**

```text
Ratnam, A. R. (2026). *An Interpretable Machine Learning Framework for River Water Hardness Classification*. Zenodo. https://doi.org/10.5281/zenodo.21315615
```
