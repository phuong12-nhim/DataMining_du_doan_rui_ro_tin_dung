# %%
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG_DIR, MODEL_DIR, REPORT_DIR  # noqa: E402
from modeling_utils import (evaluate, load_train_valid, pos_weight, predict, split_es,  # noqa: E402
                            train_with_es)
import joblib  # noqa: E402

warnings.filterwarnings("ignore")

# %%
Xtr, ytr, Xva, yva, id_va = load_train_valid()
X_fit, X_es, y_fit, y_es = split_es(Xtr, ytr)
spw = pos_weight(ytr)
print(f"fit {X_fit.shape} | early-stop {X_es.shape} | valid {Xva.shape} | scale_pos_weight = {spw:.2f}")

# %% [markdown]
# ## Huấn luyện 2 mô hình với class weight (scale_pos_weight = tỷ lệ âm/dương)
# %%
results, preds, models = [], {}, {}
for kind, label in [("xgboost", "XGBoost"), ("lightgbm", "LightGBM")]:
    t0 = time.time()
    model, best_iter = train_with_es(kind, None, X_fit, y_fit, X_es, y_es, scale_pos_weight=spw)
    p = predict(model, Xva)
    res = evaluate(f"{label} (default)", yva, p)
    res.update(best_iteration=best_iter, train_seconds=round(time.time() - t0, 1))
    results.append(res)
    preds[label], models[label] = p, model
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})

df = pd.DataFrame(results).set_index("model")
df.to_csv(REPORT_DIR / "gbm_baseline_results.csv", encoding="utf-8-sig")
print(df[["roc_auc", "pr_auc", "recall@best", "precision@best", "f1@best"]].round(4))

# %% [markdown]
# ## Feature importance (gain) + lưu kết quả
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (label, m) in zip(axes, models.items()):
    imp = pd.Series(m.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    imp.to_csv(REPORT_DIR / f"feature_importance_{label.lower()}.csv", encoding="utf-8-sig", header=["importance"])
    imp.head(20)[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"Top 20 feature importance - {label}")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_gbm_feature_importance.png", dpi=130)
plt.close()

for label, m in models.items():
    joblib.dump(m, MODEL_DIR / f"{label.lower()}_default.joblib", compress=3)
out = pd.DataFrame({"TARGET": yva.values, **{f"proba_{k.lower()}_default": v for k, v in preds.items()}})
out.insert(0, "SK_ID_CURR", id_va.values)
out.to_csv(REPORT_DIR / "gbm_baseline_valid_predictions.csv", index=False)
(REPORT_DIR / "gbm_baseline_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("Đã lưu mô hình, dự đoán và kết quả vào models/ và reports/")