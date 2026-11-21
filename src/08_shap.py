# %%
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG_DIR, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, REPORT_DIR  # noqa: E402
from modeling_utils import _clean_names, load_train_valid, predict  # noqa: E402

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--top", type=int, default=20)
args = ap.parse_args()

# %%
meta = json.loads((MODEL_DIR / "best_model.json").read_text(encoding="utf-8"))
model = joblib.load(MODEL_DIR / "best_model.joblib")
_, _, Xva, yva, _ = load_train_valid()
print(f"Mô hình giải thích: {meta['name']}")

Xs = Xva.sample(n=min(args.n, len(Xva)), random_state=RANDOM_STATE)
ys = yva.loc[Xs.index]

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xs)
if isinstance(sv, list):          # một số phiên bản shap + LightGBM trả về [lớp 0, lớp 1]
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]
base = float(np.ravel(explainer.expected_value)[-1])
print("SHAP values:", sv.shape, "| giá trị nền (log-odds):", round(base, 3))

# %% [markdown]
# ## 1. Tầm quan trọng toàn cục + hướng ảnh hưởng
# Hướng ảnh hưởng = hệ số tương quan hạng (Spearman) giữa GIÁ TRỊ đặc trưng và SHAP value:
# dương -> giá trị càng cao càng tăng rủi ro vỡ nợ; âm -> giá trị càng cao càng giảm rủi ro.
# %%
fdict_path = PROCESSED_DIR / "feature_dictionary.csv"
fd = pd.read_csv(fdict_path) if fdict_path.exists() else None
eng = {}
if fd is not None:
    # tên cột đã được làm sạch ở modeling_utils nên làm sạch tên trong dictionary theo cùng cách để khớp
    eng = dict(zip(_clean_names(fd["feature"]), fd["engineered"].tolist()))

imp = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
dirs, corrs = [], []
for j, c in enumerate(Xs.columns):
    x = Xs[c].values
    if np.std(x) == 0 or np.std(sv[:, j]) == 0:
        corrs.append(0.0)
    else:
        corrs.append(float(spearmanr(x, sv[:, j])[0]))
imp["corr_value_shap"] = corrs
imp["direction"] = np.where(imp["corr_value_shap"] > 0.1, "giá trị cao -> TĂNG rủi ro",
                    np.where(imp["corr_value_shap"] < -0.1, "giá trị cao -> GIẢM rủi ro", "không đơn điệu / phức tạp"))
imp["engineered"] = imp["feature"].map(eng) if eng else np.nan
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp["share_%"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
imp.to_csv(REPORT_DIR / "shap_importance.csv", index=False, encoding="utf-8-sig")
print(imp.head(args.top)[["feature", "mean_abs_shap", "share_%", "direction"]].round(4).to_string())

# %% [markdown]
# ## 2. Biểu đồ SHAP
# %%
plt.figure()
shap.summary_plot(sv, Xs, max_display=args.top, show=False)
plt.title(f"SHAP summary - {meta['name']}")
plt.tight_layout()
plt.savefig(FIG_DIR / "15_shap_summary.png", dpi=140, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(sv, Xs, plot_type="bar", max_display=args.top, show=False)
plt.title("Mean |SHAP| - mức độ ảnh hưởng trung bình")
plt.tight_layout()
plt.savefig(FIG_DIR / "16_shap_bar.png", dpi=140, bbox_inches="tight")
plt.close()

for f in imp["feature"].head(4):
    plt.figure()
    shap.dependence_plot(f, sv, Xs, interaction_index="auto", show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"17_shap_dependence_{f}.png", dpi=130, bbox_inches="tight")
    plt.close()

# waterfall: 1 khách hàng rủi ro cao (dự đoán cao nhất trong nhóm vỡ nợ thật) và 1 khách hàng rủi ro thấp
proba_s = predict(model, Xs)
cases = {"high_risk": int(np.argmax(np.where(ys.values == 1, proba_s, -1))),
         "low_risk": int(np.argmin(np.where(ys.values == 0, proba_s, 2)))}
for tag, i in cases.items():
    exp = shap.Explanation(values=sv[i], base_values=base, data=Xs.iloc[i].values, feature_names=list(Xs.columns))
    plt.figure()
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(f"{tag}: điểm dự đoán (chưa hiệu chuẩn)={proba_s[i]:.3f}, TARGET thực tế={int(ys.iloc[i])}")
    plt.savefig(FIG_DIR / f"18_shap_waterfall_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## 3. Bảng phân tích tự động (dùng làm khung cho phần nhận xét trong báo cáo)
# %%
lines = [f"# Phân tích SHAP - {meta['name']}\n",
         f"Tính trên {len(Xs):,} mẫu ngẫu nhiên của tập valid. Giá trị nền (log-odds) = {base:.3f}.\n",
         "| # | Đặc trưng | Mean abs SHAP | % tổng | Hướng ảnh hưởng | Tự tạo (FE) |", "|---|---|---|---|---|---|"]
for k, r in imp.head(args.top).iterrows():
    fe = "" if pd.isna(r["engineered"]) else ("có" if r["engineered"] else "không")
    lines.append(f"| {k + 1} | {r['feature']} | {r['mean_abs_shap']:.4f} | {r['share_%']:.1f}% | {r['direction']} | {fe} |")
top10 = imp.head(10)["share_%"].sum()
lines += ["", f"- 10 đặc trưng đầu đóng góp {top10:.1f}% tổng mức ảnh hưởng.",
          "- Dấu hiệu 'giá trị cao -> TĂNG rủi ro' nghĩa là khách hàng có giá trị đặc trưng lớn hơn có xác suất vỡ nợ cao hơn "
          "(ceteris paribus, theo mô hình; đây là quan hệ mô hình học được, không phải quan hệ nhân quả).",
          "- Cần tự diễn giải ý nghĩa nghiệp vụ từng đặc trưng (ví dụ EXT_SOURCE_*: điểm tín dụng từ nguồn ngoài, càng cao càng ít rủi ro)."
          " Xem thêm các biểu đồ dependence để thấy ngưỡng và tương tác."]
(REPORT_DIR / "shap_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("\nĐã lưu biểu đồ vào reports/figures/ và bảng vào reports/shap_*.{csv,md}")