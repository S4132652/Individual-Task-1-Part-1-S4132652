"""
COSC2669/COSC2816 Assignment - Part 1.3 Data Analysis
Two ML algorithms (Support Vector Machine, Neural Network/MLP) applied separately to:
  Dataset 1: PowerCo customer churn data (classification) -- SME energy-retail customers
  Dataset 2: IBM Telco customer churn data (classification) -- subscription/telecom customers
Both datasets are customer-level churn-prediction problems, chosen so that the two
data sources can be directly compared (same task, same two algorithms, different
customer base/industry) -- this supports the "are the insights complementary or
contradicting" discussion in Part 1.3 far better than an unrelated macro dataset would.
SVM was chosen (over e.g. Random Forest) specifically so that neither algorithm risks
overlapping with models used in prior coursework (e.g. Practical Data Science).
Outputs: metrics_summary.json, and plots in ./plots/
Expects input CSVs in ./data/: client_data.csv, price_data.csv, telco_churn.csv
"""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
RANDOM_STATE = 42
DATA_DIR = "data"
PLOTS_DIR = "plots"
import os
os.makedirs(PLOTS_DIR, exist_ok=True)
results = {}
def clf_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }
def run_dataset(name, X, y, features, color, plot_prefix):
    print("=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)
    print("Shape:", X.shape, "| churn rate:", round(y.mean(), 4))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # --- Model A: Support Vector Machine (RBF kernel) ---
    # SVC supports class_weight='balanced' natively, handling churn imbalance
    # the same way class_weight worked for tree models, without needing SMOTE.
    svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                  class_weight="balanced", random_state=RANDOM_STATE)
    svm_clf.fit(X_train_s, y_train)
    svm_pred = svm_clf.predict(X_test_s)
    svm_proba = svm_clf.predict_proba(X_test_s)[:, 1]
    # --- Model B: Neural Network (MLP) ---
    # MLPClassifier has no native class_weight, so we SMOTE-rebalance the
    # training set only (test set stays untouched, reflecting the real class mix).
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_s_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=RANDOM_STATE, early_stopping=True)
    mlp_clf.fit(X_train_s_bal, y_train_bal)
    mlp_pred = mlp_clf.predict(X_test_s)
    mlp_proba = mlp_clf.predict_proba(X_test_s)[:, 1]
    svm_metrics = clf_metrics(y_test, svm_pred, svm_proba)
    mlp_metrics = clf_metrics(y_test, mlp_pred, mlp_proba)
    print("SVM:", svm_metrics)
    print("MLP Neural Net:", mlp_metrics)
    results[name] = {
        "n_samples": int(len(X)),
        "n_features": len(features),
        "churn_rate": round(float(y.mean()), 4),
        "svm": svm_metrics,
        "neural_network": mlp_metrics,
    }
    # Feature importance via permutation importance (model-agnostic; SVC/RBF has
    # no native feature_importances_ the way a tree model does).
    perm = permutation_importance(svm_clf, X_test_s, y_test, scoring="f1",
                                   n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importances = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    importances[::-1].plot(kind="barh", color=color)
    plt.title(f"Top 10 Permutation Importances — SVM ({name})")
    plt.xlabel("Mean F1 drop when feature is shuffled")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{plot_prefix}_svm_feature_importance.png", dpi=150)
    plt.close()
    # ROC curves
    plt.figure(figsize=(6, 6))
    for mname, proba in [("SVM", svm_proba), ("Neural Network (MLP)", mlp_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{mname} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{plot_prefix}_roc_curve.png", dpi=150)
    plt.close()
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (mname, pred) in zip(axes, [("SVM", svm_pred), ("Neural Network", mlp_pred)]):
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(mname)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No churn", "Churn"]); ax.set_yticklabels(["No churn", "Churn"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{plot_prefix}_confusion_matrices.png", dpi=150)
    plt.close()
    return importances
# =========================================================================
# DATASET 1: PowerCo churn data
# =========================================================================
client = pd.read_csv(f"{DATA_DIR}/client_data.csv")
price = pd.read_csv(f"{DATA_DIR}/price_data.csv")
price_agg = price.groupby("id").agg(
    mean_price_off_peak_var=("price_off_peak_var", "mean"),
    mean_price_peak_var=("price_peak_var", "mean"),
    mean_price_mid_peak_var=("price_mid_peak_var", "mean"),
    mean_price_off_peak_fix=("price_off_peak_fix", "mean"),
    mean_price_peak_fix=("price_peak_fix", "mean"),
    mean_price_mid_peak_fix=("price_mid_peak_fix", "mean"),
    std_price_off_peak_var=("price_off_peak_var", "std"),
    std_price_off_peak_fix=("price_off_peak_fix", "std"),
).reset_index().fillna(0)
df = client.merge(price_agg, on="id", how="left")
for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
df["tenure_days"] = (df["date_end"] - df["date_activ"]).dt.days
df["days_to_renewal"] = (df["date_renewal"] - pd.Timestamp("2016-01-01")).dt.days
df["has_gas_flag"] = (df["has_gas"] == "t").astype(int)
cat_cols = ["channel_sales", "origin_up"]
for c in cat_cols:
    df[c] = df[c].fillna("missing")
    df[c] = LabelEncoder().fit_transform(df[c])
num_features = [
    "cons_12m", "cons_gas_12m", "cons_last_month", "forecast_cons_12m",
    "forecast_cons_year", "forecast_discount_energy", "forecast_meter_rent_12m",
    "forecast_price_energy_off_peak", "forecast_price_energy_peak",
    "forecast_price_pow_off_peak", "imp_cons", "margin_gross_pow_ele",
    "margin_net_pow_ele", "nb_prod_act", "net_margin", "num_years_antig",
    "pow_max", "tenure_days", "days_to_renewal", "has_gas_flag",
    "mean_price_off_peak_var", "mean_price_peak_var", "mean_price_mid_peak_var",
    "mean_price_off_peak_fix", "mean_price_peak_fix", "mean_price_mid_peak_fix",
    "std_price_off_peak_var", "std_price_off_peak_fix",
]
features1 = num_features + cat_cols
df[num_features] = df[num_features].apply(pd.to_numeric, errors="coerce")
df[num_features] = df[num_features].fillna(df[num_features].median())
imp1 = run_dataset("powerco_churn", df[features1], df["churn"], features1, "#2b6cb0", "churn")
# =========================================================================
# DATASET 2: IBM Telco customer churn data
# =========================================================================
telco = pd.read_csv(f"{DATA_DIR}/telco_churn.csv")
telco = telco.drop(columns=["customerID"])
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
telco["TotalCharges"] = telco["TotalCharges"].fillna(telco["TotalCharges"].median())
telco["Churn"] = (telco["Churn"] == "Yes").astype(int)
cat_cols_t = [c for c in telco.select_dtypes(exclude="number").columns if c != "Churn"]
for c in cat_cols_t:
    telco[c] = LabelEncoder().fit_transform(telco[c])
features2 = [c for c in telco.columns if c != "Churn"]
imp2 = run_dataset("telco_churn", telco[features2], telco["Churn"], features2, "#2f855a", "telco")
print()
print("Top PowerCo permutation importances:\n", imp1.head(6))
print()
print("Top Telco permutation importances:\n", imp2.head(6))
with open("metrics_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print()
print("Saved metrics_summary.json and plots to", PLOTS_DIR)
