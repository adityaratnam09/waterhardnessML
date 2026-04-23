Classification of River Water Hardness using Machine-Learning Methods

About the Project

This project presents a machine learning framework for classifying river water hardness using routine in situ physicochemical measurements. The objective is to eliminate the need for complex and reagent-heavy EDTA complexometric titration in field settings by utilizing predictive modeling.

By replacing routine laboratory titrations with a machine learning screening step, this workflow minimizes laboratory reagent consumption and supports scalable, reagent-free hardness monitoring, aligning with the principles of green chemistry and industrial water management.

Dataset

The dataset consists of 217 water samples collected from five distinct locations within the anthropogenically impacted Matanza-Riachuelo river basin in Buenos Aires province, Argentina. The samples were gathered over twelve field campaigns between May and November 2023.

The model takes the following eleven numeric physicochemical features as inputs:
	•	Ambient temperature (°C)
	•	Ambient humidity
	•	Sample water temperature (°C)
	•	pH
	•	Electrical conductivity (EC, µS/cm)
	•	Total dissolved solids (TDS, mg/L)
	•	Total suspended solids (TSS, mL sed/L)
	•	Dissolved oxygen (DO, mg/L)
	•	Water level (cm)
	•	Turbidity (NTU)
	•	Total Chloride (Cl⁻, mg/L)

The objective is to classify samples into locally defined Spanish-language categories: blanda (soft to moderately hard, <120 mg CaCO₃/L) and semidura (120–180 mg CaCO₃/L).

Data Source

The dataset used in this study is openly available on Kaggle:
	•	Dataset URL: https://www.kaggle.com/datasets/natanaelferran/river-water-parameters
	•	Source Citation: Ferran, N. (2024). River water parameters [Dataset].

Methodology & Results

Four machine learning classifiers were evaluated in the study:
	•	Random Forest (RF): Achieved the highest test accuracy of 95.45% and an AUC of 0.9950.
	•	Support Vector Machine (SVM): 79.55% test accuracy.
	•	Logistic Regression (LR): 72.73% test accuracy.
	•	K-Nearest Neighbors (KNN): 70.45% test accuracy.
  
Baseline performance levels were established using a majority-class classifier (71.4%) and a single-parameter Electrical Conductivity (EC) threshold rule (75.00%).

A feature minimization study identified that a minimal suite of just three sensors—EC, sample temperature, and TSS—matches the high predictive accuracy of the full model. This enables lower-cost, highly portable hardware deployments in the field without sacrificing accuracy.
