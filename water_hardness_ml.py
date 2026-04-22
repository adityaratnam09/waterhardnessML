# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from IPython.display import display, clear_output

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier

#analysis packages
import seaborn as sns
import shap
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_fscore_support)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import learning_curve

# ================================
# 1. Upload & Load Dataset
# ================================
# The `files` module is specific to Google Colab and often needs to be imported
# right before use if running in a non-Colab environment or if the session resets.
# For continuous execution within Colab, importing it at the top is fine, but
# keeping it near its usage for upload can sometimes prevent issues in different setups.
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)
print(f"Successfully loaded dataset: {file_name}")

# ================================
# 2. Clean Column Names
# ================================
#df.columns = df.columns.str.strip().str.replace('\n', '_')
df.columns = df.columns.str.strip().str.replace('\n', ' ')
print("Columns:", df.columns.tolist())

# ================================
# 3. Drop Non-Numeric / Irrelevant Columns
# ================================
cols_to_drop = [
    'Date (DD/MM/YYYY)',
    'Time (24 hrs XX:XX)',
    'Sampling point',
    'Hardness_classification'   # TARGET — removed from features
]

# Drop the raw hardness value to avoid data leakage.
# The classification IS just a threshold on this column, so including it
# would let the model trivially learn the rule rather than the chemistry.
hardness_col = [c for c in df.columns if 'Hardness' in c and 'classification' not in c]
cols_to_drop += hardness_col

for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

# ================================
# 4. Set Target Column
# ================================
df_full = pd.read_csv(file_name)
df_full.columns = df_full.columns.str.strip().str.replace('\n', '_')

target_col = [c for c in df_full.columns if 'classification' in c.lower()][0]
print(f"Target column: {target_col}")

y_raw = df_full[target_col].astype(str).str.lower().str.strip()

# ================================
# 5. Filter Valid Labels & Encode
# ================================
valid_mask = y_raw.isin(['blanda', 'semidura'])
df = df[valid_mask].reset_index(drop=True)
y_raw = y_raw[valid_mask].reset_index(drop=True)

print(f"\nClass distribution:\n{y_raw.value_counts()}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ================================
# 6. Class Distribution Plot
# ================================
print("\nGenerating the Class Distribution Plot...")
y_raw.value_counts().plot(kind='bar', color=['steelblue', 'salmon'], edgecolor='black')
plt.title("Class Distribution: Hardness Classification", fontsize=14, pad=20, weight='bold')
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('Class Distribution Plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 7. Select Numeric Features & Impute Missing Values
# ================================
X = df.select_dtypes(include=[np.number])
print(f"\nFeatures used ({X.shape[1]}): {X.columns.tolist()}")
print(f"\nMissing values per column:\n{X.isnull().sum()}")

# Impute with median — robust to the extreme Turbidity outliers (1000 NTU)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"\nFinal dataset size: {X.shape[0]} rows, {X.shape[1]} features")

# ================================
# 8. Train-Test Split (stratified)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 10. Baseline Model (EC Threshold)
# ================================
# EC is the chemically motivated baseline: high ionic strength (high EC)
# correlates directly with higher Ca2+ and Mg2+ concentrations and thus
# greater hardness. A simple median threshold tests whether a single
# field-readable measurement can already classify hardness reasonably.
print("\nRunning a baseline EC threshold model...")
ec_col = [c for c in X.columns if 'EC' in c][0]
ec_threshold = X_train[ec_col].median()
y_pred_baseline = [1 if val > ec_threshold else 0 for val in X_test[ec_col]]
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print(f"\nEC-Threshold Baseline Accuracy: {baseline_acc:.2%} (EC Median: {ec_threshold:.1f} µS/cm)")

# ================================
# 11. Scale Features (KNN, SVM & Logistic Regression only)
# ================================
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 12. Train & Evaluate Models
# ================================
print("\nTraining & evaluating models...")

#num of RF tress
num_rf_model_estimators = 100
rf_model = RandomForestClassifier(num_rf_model_estimators, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
lr_model = LogisticRegression(max_iter=1000)
svm_model = SVC(kernel='rbf', probability=True, random_state = 42)
#dt_model = DecisionTreeClassifier( max_depth=4, random_state=42)
#xgb_model = XGBClassifier(max_depth= 8, n_estimators= 125, random_state= 0,  learning_rate= 0.03)
#ada_model = AdaBoostClassifier(learning_rate= 0.1,n_estimators= 500,random_state=42)

models = {
    "KNN":(knn_model, X_train_scaled, X_test_scaled),
    "LR":(lr_model, X_train_scaled, X_test_scaled),
    "SVM":(svm_model, X_train_scaled, X_test_scaled),
    "RF":(rf_model, X_train, X_test),
#   "DT":(dt_model, X_train, X_test),
#   "XGB":(xgb_model, X_train, X_test),
#   "Ada":(ada_model, X_train, X_test),
}

results_list = []

# For the Combined ROC
plt.figure(figsize=(10, 8))

results = {}
for name, (model, X_tr, X_te) in models.items():

    # Ensure X_tr and X_te are DataFrames to keep feature names consistent
    if not isinstance(X_tr, pd.DataFrame):
        X_tr = pd.DataFrame(X_tr, columns=X.columns)
    if not isinstance(X_te, pd.DataFrame):
        X_te = pd.DataFrame(X_te, columns=X.columns)

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print("\n" + "="*50)
    print(f"  {name}  |  Accuracy: {acc:.2%}")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    #Collect Metrics for the Master Table
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred,
                                                               average='weighted', zero_division=0)
    results_list.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (W)": round(precision, 2),
        "Recall (W)": round(recall, 2),
        "F1-Score (W)": round(f1, 2)
    })

    # Prepare and Plot ROC Curve
    # Logic to handle predict_proba (Trees/KNN) vs decision_function (SVM/LR)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

# ================================
# 13.Consolidated ROC Curves
# ================================
print("\nGenerating Consolidated ROC Curves...")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve: Performance Comparison', fontsize=14, pad=20, weight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('Consolidated ROC Curves.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 14.Consolidated Metrics
# ================================
#df_results = pd.DataFrame(results_list).sort_values(by="Accuracy (%)", ascending=False)
df_results = pd.DataFrame(results_list)
print("\n" + "="*50)
print("       MASTER MODEL COMPARISON TABLE")
print("="*50)
print(df_results.to_string(index=False))

# ================================
# 15. Accuracy-only Model Comparison Bar Chart
# ================================
print("\nGenerating Accuracy Model Performance stats...")
plt.figure(figsize=(6, 4))
#plt.bar(results.keys(), results.values(),edgecolor='black')
plt.bar(results.keys(), results.values(),
        color=plt.cm.Set3(range(len(results))), edgecolor='black')
plt.ylim(0.5, 1.05)
plt.title("Model Accuracy Comparison", fontsize=14, pad=20, weight='bold')
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('Accuracy Model Comparison Bar Chart.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 16. Comprehensive Model Performance Bar Chart
# ================================
print("\nGenerating Comprehensive Model Performance stats...")
# 1. Prepare the plotting data (Set Model as index)
df_plot = df_results.set_index('Model').copy()

# 2. Scale Precision, Recall, and F1 to match Accuracy (0-100 scale)
for col in ['Precision (W)', 'Recall (W)', 'F1-Score (W)']:
    if col in df_plot.columns:
        df_plot[col] = df_plot[col] * 100

# 3. Generate grouped bar chart
ax = df_plot.plot(kind='bar', figsize=(15, 7), edgecolor='black', width=0.9, colormap='coolwarm')

# 4. Add data labels on top of each bar
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')

# 5. Formatting
plt.title("Comprehensive Model Performance Comparison", fontsize=14, pad=20, weight='bold')
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Machine Learning Model", fontsize=12)
plt.xticks(rotation=0)  # Keep model names horizontal
plt.ylim(0, 115)        # Leave room for labels
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Comprehensive Model Performance Bar Chart.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 16. Comprehensive Model Performance Table
# ================================
print("\nGenerating Comprehensive Model Performance Table...")
df_table = pd.DataFrame(df_results.set_index('Model'))

# Display table with styling (works best in Jupyter/Colab)
styled_table = df_table.style.background_gradient(cmap='Pastel1', axis=0)\
                             .format("{:.2f}", subset=["Precision (W)", "Recall (W)", "F1-Score (W)"])\
                             .format("{:.2f}%", subset=["Accuracy (%)"])
display(styled_table)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

#create a simple table for the report without styling
#pd.plotting.table(ax, df_table, loc='center', cellLoc='center')

#create a table for the report with styling
#use reset_index() to bring back the model names
table = ax.table(cellText=df_table.reset_index().values,
                 colLabels=df_table.reset_index().columns,
                 loc='center',
                 cellLoc='center')

# styling
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)
for (row, col), cell in table.get_celld().items():
    if row == 0: # Header
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')
        cell.set_edgecolor('black')
    else: # Body
        cell.set_facecolor('white')
        cell.set_edgecolor('#D3D3D3')

plt.savefig('Comprehensive Model Performance Table.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 17. 5-fold Stratified Cross-Validation
# ================================
print("\nGenerating 5-fold Stratified Cross-Validation...")
print("\n" + "="*50)
print("    5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*50)

cv_shuffled = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_all = {}

full_scaler = StandardScaler()
X_scaled = full_scaler.fit_transform(X)

for name, (model, _, _) in models.items():
    # Use scaled data for KNN/LR/SVM, original for others
    X_input = X_scaled if name in ["KNN", "LR", "SVM"] else X

    scores = cross_val_score(model, X_input, y, cv=cv_shuffled, scoring='accuracy')
    cv_scores_all[name] = scores
    print(f"{name:5}: Mean Acc = {scores.mean()*100:.2f}% | Std = {scores.std()*100:.2f}%")

# Generate Consolidated Stability Plot
plt.figure(figsize=(10, 6))
labels = list(cv_scores_all.keys())
data = [cv_scores_all[name] * 100 for name in labels]

# Generate the boxplot
bp = plt.boxplot(data, vert=False, patch_artist=True, tick_labels=labels,
                 medianprops=dict(color='red', linewidth=2))

# Apply coolwarm or viridis colors in two lines
colors = plt.cm.coolwarm(np.linspace(0, 1, len(labels)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Consolidated Model Stability Comparison (5-Fold CV)', fontsize=14, pad=20, weight='bold')
plt.xlabel('Accuracy (%)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Consolidated Model Stability Comparison 5-Fold CV.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ============================================================
# 18. PCA decision bounadries
# ============================================================
print("\nGenerating Decision Boundary Visualization...")

# 1. Reduce features to 2D for visualization
# We use all features (X) to ensure the PCA space covers the whole data range
pca_viz = PCA(n_components=2)
X_pca = pca_viz.fit_transform(X)

# 2. Define the models to compare using your existing parameters
# We use the parameters from our already-trained models for consistency
viz_models = {
    'KNN': KNeighborsClassifier(**knn_model.get_params()),
    'Logistic Regression': LogisticRegression(**lr_model.get_params()),
    'SVM': SVC(**svm_model.get_params()),
    'Random Forest': RandomForestClassifier(**rf_model.get_params()),
}

fig, axes = plt.subplots(1, len(viz_models), figsize=(18, 6))

for i, (name, mod) in enumerate(viz_models.items()):
    # Train the "proxy" model on 2D PCA data
    mod.fit(X_pca, y)

    # Create the decision boundary display
    DecisionBoundaryDisplay.from_estimator(
        mod, X_pca,
        response_method="predict",
        cmap='RdBu',
        alpha=0.3,
        ax=axes[i],
        xlabel="Principal Component 1",
        ylabel="Principal Component 2"
    )

    # Scatter plot of the actual data points
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                               edgecolor='white', linewidth=0.5, s=35,
                               cmap='RdBu')
    axes[i].set_title(name, fontweight='bold', fontsize=12)

plt.suptitle("Decision Boundary Comparison: RF Complexity vs. Linear Simplicity",
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()

plt.savefig('Decision Boundaries Comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# =======================================
# 19. Learning Curve (Random Forest)
# =======================================
print("\nGenerating Learning Curve - Random Forest...")
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(**rf_model.get_params()), X, y,
    cv=cv_shuffled, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.title("Learning Curve: Random Forest", fontsize=14, pad=20, fontweight='bold')
plt.xlabel("Training Examples"), plt.ylabel("Accuracy Score")
plt.legend(loc="best"), plt.grid(True)

plt.savefig('Learning Curve - Random Forest.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 20. Confusion Matrix (Random Forest)
# ================================
print("\nGenerating Confusion Matrix - Random Forest...")
y_pred_rf = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix — Random Forest", fontsize=14, pad=20, weight='bold')
plt.tight_layout()
plt.savefig('Confusion Matrix — Random Forest.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 21. Gini Feature Importance (Random Forest)
# ================================
print("\nGenerating Gini Feature Importance Study - Random Forest...")
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.barh(X.columns[sorted_idx], importances[sorted_idx], color='steelblue')
plt.title("Feature Importance — Random Forest", fontsize=14, pad=20, weight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('Feature Importance — Random Forest.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 22. Permutation Importance (Random Forest)
# ================================
print("\nGenerating Permutation Importance Study - Random Forest...")
perm_result = permutation_importance(rf_model, X_test, y_test,
                                     n_repeats=10, random_state=42)
sorted_idx = perm_result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.boxplot(perm_result.importances[sorted_idx].T, vert=False, tick_labels=X.columns[sorted_idx])
plt.title("Permutation Importance (Test Set) - Random Forest", fontsize=14, pad=20, weight='bold')
plt.xlabel("Decrease in Accuracy Score (when feature is shuffled)")
plt.tight_layout()
plt.savefig('Permutation Importance — Random Forest.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 23. SHAP Interpretability (Random Forest)
# ================================
print("\nGenerating SHAP Interpretability - Random Forest...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Version-robust SHAP selection for binary classification
if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_to_plot = shap_values[:, :, 1]
else:
    shap_to_plot = shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_to_plot, X_test, show=False)
plt.title("SHAP Feature Impact on RF Classification", fontsize=14, pad=20, weight='bold')
plt.tight_layout()
plt.savefig('SHAP Feature Impact on RF Classification.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 24. Ablation Study: Removing Turbidity (Random Forest)
# ================================
print("\nGenerating Ablation Study: Removing Turbidity - Random Forest...")
turb_col = [c for c in X.columns if 'Turbidity' in c][0]

X_train_no_turb = X_train.drop(columns=[turb_col])
X_test_no_turb = X_test.drop(columns=[turb_col])

print(f"Original feature count: {X_train.shape[1]}")
print(f"Ablated feature count:  {X_train_no_turb.shape[1]}")

rf_ablation = RandomForestClassifier(num_rf_model_estimators, random_state=42)
rf_ablation.fit(X_train_no_turb, y_train)
ablation_acc = rf_ablation.score(X_test_no_turb, y_test)

print(f"Model expected {len(rf_ablation.feature_importances_)} features.")
print(f"Features used: {list(X_train_no_turb.columns)}")

print(f"\nAblation Study - Random Forest:")
print(f"  Accuracy WITH Turbidity:    {accuracy_score(y_test, y_pred_rf):.2%}")
print(f"  Accuracy WITHOUT Turbidity: {ablation_acc:.2%}")

acc_with = accuracy_score(y_test, y_pred_rf)
acc_without = ablation_acc

labels = ['With Turbidity', 'Without Turbidity']
scores = [acc_with, acc_without]

plt.figure(figsize=(6, 5))

#apply coolwarm or viridis colors
colors = plt.cm.coolwarm([0.3, 0.7])
bars = plt.bar(labels, scores, color=colors, edgecolor='black', width=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2%}',
             ha='center', va='bottom', fontweight='bold')

plt.title("Ablation Analysis: Feature Redundancy", fontsize=14, pad=20, weight='bold')
plt.ylabel("Accuracy Score")
plt.ylim(0.5, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('Ablation Study on Random Forest using Turbidity', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 25. Inter-Feature Correlation heatmap
# ================================
print("\nGenerating Inter-Feature Correlation heatmap...")
corr_matrix = X.corr()

plt.figure(figsize=(12, 10))
# Use 'coolwarm' or 'viridis' colors for clear contrast
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Inter-Feature Correlation Matrix", fontsize=14, pad=20, weight='bold')
plt.savefig('Inter-Feature Correlation Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ============================================================
# 26. Comparative RF Ablation Analysis: Full vs. Minimalist Set with ROC curve
# ============================================================
print("\nRunning Comparative Ablation Analysis with Minimalist RF Feature Set...")
# Define the Feature Sets

# Dynamically find the columns by partial names without units
# This searches for columns containing these keywords
keywords = ['TDS', 'TSS', 'Sample temperature']
#um, let's forget we ever tried this ;)
#keywords = ['TDS', 'TSS',]
minimal_features = [next(c for c in X.columns if kw in c) for kw in keywords]

print(f"Selected Minimalist Features: {minimal_features}")
full_features = list(X.columns)

feature_sets = {
    f"Full Model ({len(full_features)} Features): ": full_features,
    f"Minimalist Model ({len(minimal_features)} Features): ": minimal_features
}

ablation_results = []

plt.figure(figsize=(8, 6))

for label, f_list in feature_sets.items():
    # Subset the data
    X_tr_sub = X_train[f_list]
    X_te_sub = X_test[f_list]

    # Initialize and fit a fresh RF model
    # Using random_state=42 ensures the comparison is fair (same splits)
    clf = RandomForestClassifier(num_rf_model_estimators, random_state=42)
    clf.fit(X_tr_sub, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_te_sub)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    ablation_results.append({
        "Configuration": label,
        "Accuracy (%)": acc * 100,
        "F1-Score": f1,
        #"Feature Count": len(f_list)
    })

    y_prob = clf.predict_proba(X_te_sub)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Full Model vs Minimal Model- Random Forest')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('ROC Curve: Full Model vs Minimal Model - Random Forest', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Create Comparison Table
df_ablation = pd.DataFrame(ablation_results)
print("\n" + "="*60)
print(df_ablation.to_string(index=False))
print("="*60)

# Visualization: Performance Decay Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

#use coolwarm or viridis
ax = sns.barplot(data=df_ablation, x='Configuration', y='Accuracy (%)',
                 hue='Configuration', palette='coolwarm', edgecolor='black')

plt.title('Impact of Feature Reduction on Random Forest Performance', fontsize=14, pad=20, weight='bold')
plt.ylim(0, 110)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('')

# Add value labels
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('Feature Reduction Comparison - Random Forest.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ================================
# 27. Interactive Prediction Widget
# ================================
# from ipywidgets import widgets, VBox, Layout
# from IPython.display import display, clear_output # Moved to top

from ipywidgets import widgets, VBox, Layout

print("\n" + "="*50)
print("           WATER HARDNESS PREDICTOR")
print("="*50)

sliders = {}
for col in X.columns:
    col_min = float(X[col].min())
    col_max = float(X[col].max())
    col_mean = float(X[col].mean())
    step = round((col_max - col_min) / 100, 3)
    if step == 0:
        step = 0.01

    sliders[col] = widgets.FloatSlider(value=round(col_mean, 2),min=round(col_min, 2),
                                       max=round(col_max, 2), step=step,
                                       description=col[:30], style={'description_width': '200px'},
                                       layout=Layout(width='500px'))

predict_button = widgets.Button(description='Predict Hardness Class', button_style='primary',
                                layout=Layout(width='220px', height='40px'))

output = widgets.Output()

def on_predict(b):
    with output:
        clear_output()
        input_vals = pd.DataFrame([[sliders[col].value for col in X.columns]],
                                  columns=X.columns)

        pred_label = le.inverse_transform(rf_model.predict(input_vals))[0]
        proba = rf_model.predict_proba(input_vals)[0]

        eps = 1e-2  # tolerance
        if abs(proba[0] - 0.5) < eps:
          print("\n  Prediction: UNCERTAIN (near 50-50 split)")
        else:
          confidence = max(proba) * 100
          color = '\033[92m' if pred_label == 'blanda' else '\033[94m'
          reset = '\033[0m'
          print(f"\n  Predicted class : {color}{pred_label.upper()}{reset}")
          print(f"  Confidence      : {confidence:.1f}%")
          print(f"  (blanda={proba[list(le.classes_).index('blanda')]:.2f})   | "
                f"  (semidura={proba[list(le.classes_).index('semidura')]:.2f})")
          if pred_label == 'blanda':
              print("\n  Blanda = soft water  (<120 mg CaCO3/L)")
          else:
              print("\n  Semidura = moderately hard water  (120-180 mg CaCO3/L)")

predict_button.on_click(on_predict)

display(VBox(
    list(sliders.values()) + [predict_button, output],
    layout=Layout(padding='10px')
))
