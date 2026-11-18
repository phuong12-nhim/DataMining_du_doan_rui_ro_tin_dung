# %%
import argparse
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG_DIR, MODEL_DIR, REPORT_DIR  # noqa: E402
from modeling_utils import (DEFAULT_PARAMS, evaluate, load_train_valid, pos_weight, predict,  # noqa: E402
                            split_es, train_with_es)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--select", choices=["roc_auc", "pr_auc"], default="roc_auc")
args = ap.parse_args()

# %%
Xtr, ytr, Xva, yva, id_va = load_train_valid()
X_fit, X_es, y_fit, y_es = split_es(Xtr, ytr)
ratio = pos_weight(ytr)

rows, probas = [], {}

# 1) Baseline từ 03 (LogReg, RF) + GBM mặc định từ 04
for f, tag in [("baseline_results.csv", None), ("gbm_baseline_results.csv", None)]:
    p = REPORT_DIR / f
    if p.exists():
        d = pd.read_csv(p).rename(columns={"Unnamed: 0": "model"})
        d["model"] = d["model"].replace({"LogisticRegression": "Logistic Regression", "RandomForest": "Random Forest"})
        rows += d.to_dict("records")
    else:
        print(f"Bỏ qua {f} (chưa có)")
for f, cols in [("baseline_valid_predictions.csv", {"proba_logreg": "Logistic Regression", "proba_rf": "Random Forest"}),
                ("gbm_baseline_valid_predictions.csv", {"proba_xgboost_default": "XGBoost (default)",
                                                         "proba_lightgbm_default": "LightGBM (default)"})]:
    p = REPORT_DIR / f
    if p.exists():
        d = pd.read_csv(p)
        probas.update({name: d[c].values for c, name in cols.items() if c in d})

# 2) Bản đã tuning
tuned_models = {}
for kind, label in [("xgboost", "XGBoost"), ("lightgbm", "LightGBM")]:
    jp = REPORT_DIR / f"tuning_{kind}_best.json"
    if jp.exists():
        cfg = json.loads(jp.read_text(encoding="utf-8"))
        params = cfg["params"]
        spw = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": ratio}[cfg["spw_mode"]]
        print(f"{label}: dùng tham số tuning (CV {cfg['metric']}={cfg['cv_score']:.4f}, spw_mode={cfg['spw_mode']})")
    else:
        print(f"{label}: chưa có {jp.name} -> dùng tham số mặc định (hãy chạy 06_tune.py)")
        params, spw = DEFAULT_PARAMS[kind], ratio
    t0 = time.time()
    model, best_iter = train_with_es(kind, params, X_fit, y_fit, X_es, y_es, scale_pos_weight=spw)
    name = f"{label} (tuned)"
    p = predict(model, Xva)
    res = evaluate(name, yva, p)
    res.update(best_iteration=best_iter, train_seconds=round(time.time() - t0, 1))
    rows.append(res)
    probas[name] = p
    tuned_models[name] = model
    joblib.dump(model, MODEL_DIR / f"{kind}_tuned.joblib", compress=3)

# %% [markdown]
# ## Bảng so sánh
# %%
cols = ["roc_auc", "pr_auc", "precision@0.5", "recall@0.5", "f1@0.5", "best_f1_threshold",
        "precision@best", "recall@best", "f1@best", "train_seconds"]
comp = pd.DataFrame(rows).drop_duplicates("model", keep="last").set_index("model")
comp = comp[[c for c in cols if c in comp]].sort_values(args.select, ascending=False)
comp.round(4).to_csv(REPORT_DIR / "model_comparison.csv", encoding="utf-8-sig")


def md_table(df):
    head = "| Mô hình | " + " | ".join(df.columns) + " |\n|" + "---|" * (len(df.columns) + 1) + "\n"
    body = "".join(f"| {i} | " + " | ".join("" if pd.isna(v) else f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
                                          for v in r) + " |\n" for i, r in df.iterrows())
    return head + body


(REPORT_DIR / "model_comparison.md").write_text(
    "# Bảng so sánh các thuật toán (tập valid)\n\n"
    "`@0.5`: ngưỡng mặc định 0.5 | `@best`: ngưỡng tối ưu F1 (chọn trên valid, chỉ để tham khảo).\n\n"
    + md_table(comp.round(4)), encoding="utf-8")
pd.set_option("display.width", 200)
print(comp[["roc_auc", "pr_auc", "precision@best", "recall@best", "f1@best"]].round(4))

# %% [markdown]
# ## Đường cong ROC / PR và chọn mô hình tốt nhất
# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(yva, p)
    axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(yva, p):.3f})")
    pr, rc, _ = precision_recall_curve(yva, p)
    axes[1].plot(rc, pr, label=f"{name} ({average_precision_score(yva, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (AUC)")
axes[1].axhline(yva.mean(), color="k", ls="--", lw=1)
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (AP)")
for ax in axes:
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "13_model_roc_pr.png", dpi=130)
plt.close()

ax = comp[["roc_auc", "pr_auc", "f1@best", "recall@best", "precision@best"]].plot.barh(figsize=(10, 5.5), width=0.8)
ax.invert_yaxis()
ax.set_title("So sánh các mô hình trên tập valid")
plt.tight_layout()
plt.savefig(FIG_DIR / "14_model_metrics_bar.png", dpi=130)
plt.close()

cand = [n for n in comp.index if n in tuned_models]
best_name = comp.loc[cand, args.select].idxmax()
kind = "xgboost" if best_name.startswith("XGBoost") else "lightgbm"
joblib.dump(tuned_models[best_name], MODEL_DIR / "best_model.joblib", compress=3)
(MODEL_DIR / "best_model.json").write_text(json.dumps(
    {"name": best_name, "kind": kind, "select_by": args.select,
     "roc_auc": float(comp.loc[best_name, "roc_auc"]), "pr_auc": float(comp.loc[best_name, "pr_auc"])},
    indent=2), encoding="utf-8")
pd.DataFrame({"SK_ID_CURR": id_va.values, "TARGET": yva.values,
              **{f"proba_{k}": v for k, v in probas.items()}}).to_csv(REPORT_DIR / "all_valid_predictions.csv", index=False)
print(f"\nMô hình tốt nhất (theo {args.select}): {best_name} -> models/best_model.joblib")
