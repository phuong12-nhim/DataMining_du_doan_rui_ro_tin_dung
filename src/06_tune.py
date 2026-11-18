# %%
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_STATE, REPORT_DIR  # noqa: E402
from modeling_utils import load_train_valid, pos_weight, predict, train_with_es  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["xgboost", "lightgbm", "both"], default="both")
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--metric", choices=["roc_auc", "pr_auc"], default="roc_auc")
ap.add_argument("--frac", type=float, default=1.0, help="tỷ lệ train dùng khi tuning (0-1), giảm để chạy nhanh")
ap.add_argument("--timeout", type=int, default=None, help="giây tối đa cho mỗi mô hình")
args = ap.parse_args()

# %%
Xtr, ytr, _, _, _ = load_train_valid()
if args.frac < 1.0:
    keep = Xtr.sample(frac=args.frac, random_state=RANDOM_STATE).index
    Xtr, ytr = Xtr.loc[keep], ytr.loc[keep]
Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
ratio = pos_weight(ytr)
print(f"Tuning trên {Xtr.shape}, tỷ lệ âm/dương = {ratio:.2f}, metric = {args.metric}")
score_fn = roc_auc_score if args.metric == "roc_auc" else average_precision_score
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)


def suggest(trial, kind):
    spw_choice = trial.suggest_categorical("spw_mode", ["none", "sqrt", "full"])
    spw = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": ratio}[spw_choice]
    if kind == "lightgbm":
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        )
    else:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    return params, spw


def make_objective(kind):
    def objective(trial):
        params, spw = suggest(trial, kind)
        scores, iters = [], []
        for i, (a, b) in enumerate(skf.split(Xtr, ytr)):
            model, best_iter = train_with_es(kind, params, Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], ytr.iloc[b], spw)
            scores.append(score_fn(ytr.iloc[b], predict(model, Xtr.iloc[b])))
            iters.append(best_iter)
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iteration", int(np.mean(iters)))
        return float(np.mean(scores))
    return objective


# %%
kinds = ["xgboost", "lightgbm"] if args.model == "both" else [args.model]
for kind in kinds:
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))

    def log_cb(st, tr):
        if tr.state == optuna.trial.TrialState.COMPLETE:
            print(f"[{kind}] trial {tr.number:>3}  CV {args.metric}={tr.value:.4f}  (best={st.best_value:.4f})  "
                  f"{time.time() - t0:.0f}s")

    study.optimize(make_objective(kind), n_trials=args.trials, timeout=args.timeout, callbacks=[log_cb])

    bp = dict(study.best_params)
    spw_mode = bp.pop("spw_mode")
    if kind == "lightgbm":
        bp["subsample_freq"] = 1
    out = {"model": kind, "metric": args.metric, "cv_score": study.best_value, "folds": args.folds,
           "n_trials": len(study.trials), "frac_train": args.frac, "spw_mode": spw_mode,
           "best_iteration_cv": study.best_trial.user_attrs.get("best_iteration"), "params": bp}
    (REPORT_DIR / f"tuning_{kind}_best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(REPORT_DIR / f"tuning_{kind}_trials.csv", index=False)
    print(f"\n[{kind}] tốt nhất CV {args.metric} = {study.best_value:.4f}, spw_mode = {spw_mode}\n{bp}\n")