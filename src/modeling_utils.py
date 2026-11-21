import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, PROCESSED_DIR, RANDOM_STATE, TARGET  # noqa: E402

ES_ROUNDS = 100          # early stopping
ES_FRACTION = 0.10       # tách 10% train làm tập early stopping (valid giữ nguyên để đánh giá cuối)

DEFAULT_PARAMS = {
    "xgboost": dict(n_estimators=3000, learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, gamma=0.0),
    "lightgbm": dict(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0),
}

# Dữ liệu
def _read(name):
    for ext in ("parquet", "csv.gz"):
        p = PROCESSED_DIR / f"{name}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"Không thấy {name} trong {PROCESSED_DIR}. Hãy chạy 02_preprocess.py trước.")


def _clean_names(cols):
    """LightGBM/XGBoost không chấp nhận ký tự đặc biệt trong tên cột (khoảng trắng, [ ] < , : ...)."""
    out, seen = [], {}
    for c in cols:
        n = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_") or "col"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return out


def load_train_valid():
    """Trả về Xtr, ytr, Xva, yva, id_va. Dùng file *_processed (chưa scale - phù hợp cho mô hình cây)."""
    tr, va = _read("train_processed"), _read("valid_processed")
    Xtr, ytr = tr.drop(columns=[ID_COL, TARGET]), tr[TARGET].astype(int)
    Xva, yva = va.drop(columns=[ID_COL, TARGET]), va[TARGET].astype(int)
    names = _clean_names(Xtr.columns)
    Xtr.columns, Xva.columns = names, names
    return Xtr.astype("float32"), ytr, Xva.astype("float32"), yva, va[ID_COL]


def split_es(X, y):
    """Tách train thành (fit, early-stopping). Stratify theo TARGET."""
    return train_test_split(X, y, test_size=ES_FRACTION, stratify=y, random_state=RANDOM_STATE)


def pos_weight(y):
    """Tỷ lệ âm/dương (~11.4) - dùng cho scale_pos_weight."""
    return float((y == 0).sum() / max((y == 1).sum(), 1))

# Đánh giá 
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
        "recall@0.5": recall_score(y_true, pred05, zero_division=0),
        "f1@0.5": f1_score(y_true, pred05),
        "best_f1_threshold": best_thr,
        "precision@best": precision_score(y_true, predb, zero_division=0),
        "recall@best": recall_score(y_true, predb, zero_division=0),
        "f1@best": f1_score(y_true, predb),
        "tn_fp_fn_tp@best": confusion_matrix(y_true, predb, labels=[0, 1]).ravel().tolist(),
    }


def evaluate_at_threshold(name, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "threshold": float(threshold),
        "precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, pred, zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

# Mô hình
def build_model(kind, params=None, scale_pos_weight=1.0, es=True):
    """kind: 'xgboost' | 'lightgbm'. es=True -> bật early stopping (cần eval_set khi fit)."""
    p = dict(DEFAULT_PARAMS[kind])
    p.update(params or {})
    if kind == "xgboost":
        from xgboost import XGBClassifier
        if es:
            p["early_stopping_rounds"] = ES_ROUNDS
        return XGBClassifier(tree_method="hist", eval_metric="auc", scale_pos_weight=scale_pos_weight,
                             n_jobs=-1, random_state=RANDOM_STATE, **p)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="binary", metric="auc", scale_pos_weight=scale_pos_weight, n_jobs=-1,
                      random_state=RANDOM_STATE, verbosity=-1, **p)
    raise ValueError(kind)


def train_with_es(kind, params, X_fit, y_fit, X_es, y_es, scale_pos_weight=1.0):
    """Huấn luyện với early stopping trên (X_es, y_es). Trả về (model, best_iteration)."""
    model = build_model(kind, params, scale_pos_weight)
    if kind == "xgboost":
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        best = int(model.best_iteration) + 1
    else:
        import lightgbm as lgb
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        best = int(model.best_iteration_)
    return model, best


def predict(model, X):
    return model.predict_proba(X)[:, 1]
