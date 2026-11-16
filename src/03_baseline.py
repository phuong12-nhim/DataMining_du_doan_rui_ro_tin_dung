import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG_DIR, ID_COL, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, REPORT_DIR, TARGET  # noqa: E402

warnings.filterwarnings("ignore")


def load(name):
    for ext in ("parquet", "csv.gz"):
        p = PROCESSED_DIR / f"{name}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"Không thấy {name} trong {PROCESSED_DIR}. Hãy chạy 02_preprocess.py trước.")


def split_xy(d):
    return d.drop(columns=[ID_COL, TARGET]), d[TARGET]


# %% [markdown]
# ## 1. Đọc dữ liệu đã xử lý
# %%
Xtr_s, ytr = split_xy(load("train_scaled"))
Xva_s, yva = split_xy(load("valid_scaled"))
Xtr, _ = split_xy(load("train_processed"))
Xva, _ = split_xy(load("valid_processed"))
print(f"train {Xtr.shape} | valid {Xva.shape} | TARGET=1: {ytr.mean():.4f}")


# %% [markdown]
# ## 2. Hàm đánh giá
# Ngoài ngưỡng 0.5 mặc định, báo cáo thêm ngưỡng tối ưu F1 (chọn trên valid chỉ để tham khảo).
# %%
def evaluate(name, y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.nanargmax(f1s[:-1]))
    best_thr = float(thr[best])
    pred05 = (proba >= 0.5).astype(int)
    predb = (proba >= best_thr).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision@0.5": precision_score(y_true, pred05, zero_division=0),
        "recall@0.5": recall_score(y_true, pred05),
        "f1@0.5": f1_score(y_true, pred05),
        "best_f1_threshold": best_thr,
        "precision@best": precision_score(y_true, predb, zero_division=0),
        "recall@best": recall_score(y_true, predb),
        "f1@best": f1_score(y_true, predb),
        "tn_fp_fn_tp@best": confusion_matrix(y_true, predb).ravel().tolist(),
    }


# %% [markdown]
# ## 3. Logistic Regression (baseline)
# %%
t0 = time.time()
lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, solver="lbfgs",
                        random_state=RANDOM_STATE, n_jobs=-1)
lr.fit(Xtr_s, ytr)
p_lr = lr.predict_proba(Xva_s)[:, 1]
res_lr = evaluate("LogisticRegression", yva, p_lr)
res_lr["train_seconds"] = round(time.time() - t0, 1)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res_lr.items()})

# %% [markdown]
# ## 4. Random Forest
# %%
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50, max_features="sqrt",
                            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(Xtr, ytr)
p_rf = rf.predict_proba(Xva)[:, 1]
res_rf = evaluate("RandomForest", yva, p_rf)
res_rf["train_seconds"] = round(time.time() - t0, 1)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res_rf.items()})

# %% [markdown]
# ## 5. So sánh, feature importance, đường cong ROC / PR
# %%
results = pd.DataFrame([res_lr, res_rf]).set_index("model")
results.to_csv(REPORT_DIR / "baseline_results.csv", encoding="utf-8-sig")
print(results[["roc_auc", "pr_auc", "recall@best", "precision@best", "f1@best"]].round(4))

imp_rf = pd.Series(rf.feature_importances_, index=Xtr.columns, name="importance").sort_values(ascending=False)
imp_lr = pd.Series(lr.coef_[0], index=Xtr_s.columns, name="coef").sort_values(key=np.abs, ascending=False)
imp_rf.to_csv(REPORT_DIR / "feature_importance_rf.csv", encoding="utf-8-sig")
imp_lr.to_csv(REPORT_DIR / "feature_coef_logreg.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, p in [("Logistic Regression", p_lr), ("Random Forest", p_rf)]:
    fpr, tpr, _ = roc_curve(yva, p)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yva, p):.3f})")
    pr, rc, _ = precision_recall_curve(yva, p)
    axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(yva, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
axes[1].axhline(yva.mean(), color="k", ls="--", lw=1, label="Ngẫu nhiên")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "09_baseline_roc_pr.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
imp_rf.head(20)[::-1].plot.barh(ax=ax, color="#4C72B0")
ax.set_title("Top 20 feature importance - Random Forest")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_rf_feature_importance.png", dpi=130)
plt.close()

# %% [markdown]
# ## 6. Lưu mô hình + dự đoán trên valid (để nhóm khác so sánh / ensemble)
# %%
joblib.dump(lr, MODEL_DIR / "logreg_baseline.joblib")
joblib.dump(rf, MODEL_DIR / "rf_baseline.joblib", compress=3)
pd.DataFrame({ID_COL: load("valid_processed")[ID_COL], TARGET: yva.values,
              "proba_logreg": p_lr, "proba_rf": p_rf}).to_csv(
    REPORT_DIR / "baseline_valid_predictions.csv", index=False)
(REPORT_DIR / "baseline_results.json").write_text(
    json.dumps({"logreg": res_lr, "random_forest": res_rf}, indent=2, ensure_ascii=False), encoding="utf-8")
print("Đã lưu mô hình, dự đoán và kết quả vào models/ và reports/")
