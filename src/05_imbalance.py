# %%
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG_DIR, RANDOM_STATE, REPORT_DIR  # noqa: E402
from modeling_utils import (evaluate, load_train_valid, pos_weight, predict, split_es,  # noqa: E402
                            train_with_es)

warnings.filterwarnings("ignore")

# %%
Xtr, ytr, Xva, yva, _ = load_train_valid()
X_fit, X_es, y_fit, y_es = split_es(Xtr, ytr)
ratio = pos_weight(ytr)

def undersample(X, y, neg_per_pos=3):
    rng = np.random.RandomState(RANDOM_STATE)
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]
    neg_keep = rng.choice(neg_idx, size=min(len(neg_idx), neg_per_pos * len(pos_idx)), replace=False)
    idx = np.sort(np.concatenate([pos_idx, neg_keep]))
    return X.iloc[idx], y.iloc[idx]


def smote(X, y, strategy=0.3):
    from imblearn.over_sampling import SMOTE
    Xr, yr = SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=RANDOM_STATE).fit_resample(X, y)
    return Xr.astype("float32"), yr


# (tên, hàm tạo dữ liệu train, scale_pos_weight)
strategies = [
    ("none", lambda: (X_fit, y_fit), 1.0),
    ("sqrt_weight", lambda: (X_fit, y_fit), float(np.sqrt(ratio))),
    ("class_weight", lambda: (X_fit, y_fit), ratio),
    ("undersample", lambda: undersample(X_fit, y_fit), 1.0),
    ("smote", lambda: smote(X_fit, y_fit), 1.0),
]
try:
    import imblearn  # noqa: F401
except ImportError:
    print("Không có imbalanced-learn -> bỏ qua SMOTE (pip install imbalanced-learn)")
    strategies = [s for s in strategies if s[0] != "smote"]

# %%
rows = []
for kind, label in [("lightgbm", "LightGBM"), ("xgboost", "XGBoost")]:
    for name, make, spw in strategies:
        t0 = time.time()
        Xf, yf = make()
        model, best_iter = train_with_es(kind, None, Xf, yf, X_es, y_es, scale_pos_weight=spw)
        res = evaluate(f"{label} | {name}", yva, predict(model, Xva))
        res.update(algorithm=label, strategy=name, n_train=len(yf), pos_rate_train=round(float(yf.mean()), 4),
                   scale_pos_weight=round(spw, 2), best_iteration=best_iter, train_seconds=round(time.time() - t0, 1))
        rows.append(res)
        print(f"{res['model']:<28} ROC-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f} "
              f"Recall@best={res['recall@best']:.3f} Prec@best={res['precision@best']:.3f} F1@best={res['f1@best']:.4f}")

df = pd.DataFrame(rows).set_index("model")
df.to_csv(REPORT_DIR / "imbalance_results.csv", encoding="utf-8-sig")

# %% [markdown]
# ## Biểu đồ so sánh
# %%
metrics = ["roc_auc", "pr_auc", "f1@best", "recall@best"]
fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
for ax, m in zip(axes, metrics):
    pv = df.pivot(index="strategy", columns="algorithm", values=m).loc[[s[0] for s in strategies]]
    pv.plot.bar(ax=ax, rot=30, width=0.8)
    ax.set_title(m)
    ax.set_xlabel("")
    lo = max(0, pv.min().min() - 0.02)
    ax.set_ylim(lo, pv.max().max() + 0.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "12_imbalance_comparison.png", dpi=130)
plt.close()

best = df.sort_values("pr_auc", ascending=False).groupby("algorithm").head(1)
print("\nChiến lược tốt nhất theo PR-AUC cho từng thuật toán:")
print(best[["strategy", "roc_auc", "pr_auc", "f1@best"]].round(4))
print("Lưu ý: ROC-AUC/PR-AUC chỉ đo khả năng xếp hạng nên ít đổi theo class weight;"
      " Recall/Precision/F1 ở ngưỡng 0.5 mới thay đổi rõ (xem cột *@0.5 trong imbalance_results.csv).")