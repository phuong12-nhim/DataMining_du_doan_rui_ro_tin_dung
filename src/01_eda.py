import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG_DIR, ID_COL, RAW_FILE, REPORT_DIR, TARGET  # noqa: E402
from features import add_features, clean_raw  # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}  # 0 = trả đúng hạn, 1 = có khó khăn thanh toán

# Các cột số quan trọng dùng để vẽ phân phối / boxplot / so sánh theo TARGET
KEY_NUMERIC = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
]


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close()


def save_table(df, name):
    df.to_csv(REPORT_DIR / f"{name}.csv", index=True, encoding="utf-8-sig")


# %% [markdown]
# ## 0. Đọc dữ liệu
# %%
if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Không thấy {RAW_FILE}. Tải application_train.csv từ Kaggle và đặt vào data/raw/ (xem README)."
    )
df = pd.read_csv(RAW_FILE)
print("Đã đọc:", RAW_FILE.name)

# %% [markdown]
# ## 1. Kích thước dữ liệu (số dòng, số cột, dung lượng)
# %%
mem_mb = df.memory_usage(deep=True).sum() / 1024**2
size_info = pd.Series({
    "so_dong": df.shape[0],
    "so_cot": df.shape[1],
    "dung_luong_RAM_MB": round(mem_mb, 1),
    "dung_luong_file_MB": round(RAW_FILE.stat().st_size / 1024**2, 1),
    "so_dong_trung_ID": int(df[ID_COL].duplicated().sum()),
    "so_dong_trung_toan_bo": int(df.duplicated().sum()),
}, name="gia_tri")
save_table(size_info.to_frame(), "01_size")
print(size_info, "\n")

# %% [markdown]
# ## 2. Kiểu dữ liệu (numerical vs categorical)
# Cột số chỉ có 2 giá trị (0/1) là cờ nhị phân (FLAG_*), coi như categorical khi phân tích.
# %%
feature_cols = [c for c in df.columns if c not in (TARGET, ID_COL)]
cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]  # pandas 3: chuỗi là dtype "str", không phải object
num_cols = [c for c in feature_cols if c not in cat_cols]
binary_num = [c for c in num_cols if df[c].dropna().nunique() <= 2]
cont_num = [c for c in num_cols if c not in binary_num]

dtype_table = pd.DataFrame({
    "dtype": df[feature_cols].dtypes.astype(str),
    "nhom": ["categorical" if c in cat_cols else ("binary_flag" if c in binary_num else "numerical")
             for c in feature_cols],
    "n_unique": df[feature_cols].nunique(),
    "ti_le_thieu_%": (df[feature_cols].isna().mean() * 100).round(2),
})
save_table(dtype_table, "02_dtypes")
print(dtype_table["nhom"].value_counts(), "\n")

# %% [markdown]
# ## 3. Phân phối TARGET (countplot + tỷ lệ phần trăm)
# %%
vc = df[TARGET].value_counts().sort_index()
pct = (vc / vc.sum() * 100).round(2)
imbalance = vc[0] / vc[1]
target_table = pd.DataFrame({"so_luong": vc, "ti_le_%": pct})
target_table.loc["ti_le_mat_can_bang (0:1)"] = [round(imbalance, 2), np.nan]
save_table(target_table, "03_target_distribution")
print(target_table, "\n")

fig, ax = plt.subplots(figsize=(5.5, 4))
sns.countplot(x=TARGET, hue=TARGET, data=df, palette=[PALETTE[0], PALETTE[1]], legend=False, ax=ax)
for p, v in zip(ax.patches, pct):
    ax.annotate(f"{int(p.get_height()):,}\n({v}%)", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
ax.set_title("Phân phối TARGET (1 = có khó khăn thanh toán)")
ax.set_ylim(0, vc.max() * 1.15)
save_fig("03_target_countplot")

# %% [markdown]
# ## 4. Missing values (tỷ lệ thiếu theo từng cột)
# %%
miss = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
miss = miss[miss > 0].to_frame("ti_le_thieu_%")
miss["so_dong_thieu"] = df[miss.index].isna().sum()
save_table(miss, "04_missing_values")
print(f"{len(miss)}/{df.shape[1]} cột có missing; {(miss['ti_le_thieu_%'] > 60).sum()} cột thiếu > 60%\n")

top = miss.head(40)
fig, ax = plt.subplots(figsize=(8, 10))
sns.barplot(x=top["ti_le_thieu_%"], y=top.index, color="#4C72B0", ax=ax)
ax.axvline(60, color="red", ls="--", lw=1, label="Ngưỡng bỏ cột (60%)")
ax.set_xlabel("Tỷ lệ thiếu (%)")
ax.set_title("Top 40 cột có tỷ lệ missing cao nhất")
ax.legend()
save_fig("04_missing_top40")

# %% [markdown]
# ## 5. Phân phối thuộc tính (Histogram + KDE)
# DAYS_* đổi sang năm cho dễ đọc. DAYS_EMPLOYED = 365243 là giá trị bất thường nên loại ra khi vẽ.
# %%
skew = df[cont_num].skew().round(2).sort_values(key=np.abs, ascending=False).to_frame("skewness")
save_table(skew, "05_skewness")


def to_plot_series(col):
    s = df[col].copy()
    if col == "DAYS_EMPLOYED":
        s = s.where(s != 365243)
    if col.startswith("DAYS_"):
        s = -s / 365.25
        label = col.replace("DAYS_", "") + " (năm)"
    else:
        label = col
    return s.dropna(), label


plot_cols = [c for c in KEY_NUMERIC if c in df.columns]
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, col in zip(axes.ravel(), plot_cols):
    s, label = to_plot_series(col)
    s = s.clip(upper=s.quantile(0.995))  # cắt đuôi cực phải để nhìn rõ thân phân phối
    sns.histplot(s, kde=True, bins=50, color="#4C72B0", ax=ax)
    ax.set_title(f"{label}\nskew = {df[col].skew():.2f}", fontsize=10)
    ax.set_xlabel("")
for ax in axes.ravel()[len(plot_cols):]:
    ax.axis("off")
fig.suptitle("Histogram + KDE các thuộc tính số quan trọng", y=1.0, fontsize=14)
save_fig("05_hist_kde_key_numeric")

# Biến phân loại: tần suất của các giá trị
cat_plot = [c for c in ("NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE") if c in df.columns]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.ravel(), cat_plot):
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order, color="#4C72B0", ax=ax)
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("")
for ax in axes.ravel()[len(cat_plot):]:
    ax.axis("off")
save_fig("05_countplot_key_categorical")

# %% [markdown]
# ## 6. Dữ liệu ngoại lệ (Boxplot + quy tắc IQR)
# Outlier theo IQR: giá trị < Q1 - 1.5*IQR hoặc > Q3 + 1.5*IQR.
# %%
rows = []
for col in cont_num:
    s = df[col].dropna()
    if col == "DAYS_EMPLOYED":
        s = s[s != 365243]
    if s.empty:
        continue
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    rows.append({"cot": col, "Q1": q1, "Q3": q3, "IQR": iqr, "can_duoi": lo, "can_tren": hi,
                 "min": s.min(), "max": s.max(), "so_outlier": n_out,
                 "ti_le_outlier_%": round(n_out / len(s) * 100, 2)})
iqr_table = pd.DataFrame(rows).set_index("cot").sort_values("ti_le_outlier_%", ascending=False)
save_table(iqr_table, "06_outliers_iqr")

anom = int((df["DAYS_EMPLOYED"] == 365243).sum()) if "DAYS_EMPLOYED" in df else 0
print(f"DAYS_EMPLOYED = 365243: {anom:,} dòng ({anom / len(df) * 100:.1f}%) -> xử lý ở bước tiền xử lý\n")

fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, col in zip(axes.ravel(), plot_cols):
    s, label = to_plot_series(col)
    sns.boxplot(x=s, color="#8172B3", fliersize=1.5, ax=ax)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("")
for ax in axes.ravel()[len(plot_cols):]:
    ax.axis("off")
fig.suptitle("Boxplot các thuộc tính số quan trọng", y=1.0, fontsize=14)
save_fig("06_boxplot_key_numeric")

# %% [markdown]
# ## 7. Tương quan (Correlation Heatmap)
# Tính trên dữ liệu đã clean + feature engineering để biết đặc trưng mới có đóng góp gì không.
# %%
df_fe = add_features(clean_raw(df))
num_fe = df_fe.select_dtypes(include=[np.number]).drop(columns=[ID_COL], errors="ignore")
corr_target = num_fe.corr()[TARGET].drop(TARGET)
corr_rank = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index).to_frame("corr_voi_TARGET")
corr_rank["la_feature_moi"] = [c not in df.columns for c in corr_rank.index]
save_table(corr_rank, "07_corr_with_target")
print(corr_rank.head(15), "\n")

top_feats = corr_rank.head(20).index.tolist()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(num_fe[top_feats + [TARGET]].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation Heatmap - 20 đặc trưng tương quan mạnh nhất với TARGET")
save_fig("07_corr_heatmap_top20")

# Cặp đặc trưng tương quan rất cao với nhau (|r| > 0.95) -> ứng viên loại bỏ để giảm đa cộng tuyến
c = num_fe.drop(columns=[TARGET]).corr().abs()
upper = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1))
pairs = upper.stack().reset_index()
pairs.columns = ["feature_1", "feature_2", "abs_corr"]
pairs = pairs[pairs["abs_corr"] > 0.95].sort_values("abs_corr", ascending=False)
save_table(pairs.set_index("feature_1"), "07_highly_correlated_pairs")
print(f"{len(pairs)} cặp đặc trưng có |corr| > 0.95\n")

# %% [markdown]
# ## 8. Phân tích theo TARGET (so sánh đặc trưng giữa hai nhóm)
# %%
# 8a. Biến số: so sánh trung vị + KDE chồng nhau
grp = df_fe.groupby(TARGET)
cmp_rows = []
for col in num_fe.columns.drop(TARGET):
    a, b = df_fe.loc[df_fe[TARGET] == 0, col].dropna(), df_fe.loc[df_fe[TARGET] == 1, col].dropna()
    if len(a) == 0 or len(b) == 0:
        continue
    pooled = np.sqrt((a.var() + b.var()) / 2)
    cmp_rows.append({"cot": col, "mean_T0": a.mean(), "mean_T1": b.mean(),
                     "median_T0": a.median(), "median_T1": b.median(),
                     "cohen_d": (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan})
cmp_table = pd.DataFrame(cmp_rows).set_index("cot")
cmp_table = cmp_table.reindex(cmp_table["cohen_d"].abs().sort_values(ascending=False).index)
save_table(cmp_table, "08_numeric_by_target")

kde_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AGE_YEARS", "EMPLOYED_YEARS",
            "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM"]
kde_cols = [c for c in kde_cols if c in df_fe.columns]
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, col in zip(axes.ravel(), kde_cols):
    d = df_fe[[col, TARGET]].dropna()
    d = d[d[col] <= d[col].quantile(0.995)]
    sns.kdeplot(data=d, x=col, hue=TARGET, palette=PALETTE, common_norm=False, fill=True, alpha=0.3, ax=ax)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
for ax in axes.ravel()[len(kde_cols):]:
    ax.axis("off")
fig.suptitle("So sánh phân phối giữa nhóm TARGET=0 và TARGET=1", y=1.0, fontsize=14)
save_fig("08_kde_by_target")

# 8b. Biến phân loại: tỷ lệ vỡ nợ theo từng nhóm
cat_rows = []
for col in cat_cols:
    t = df.groupby(col, dropna=False)[TARGET].agg(["count", "mean"])
    t.columns = ["so_luong", "ti_le_TARGET1"]
    t["cot"] = col
    cat_rows.append(t.reset_index().rename(columns={col: "gia_tri"}))
cat_table = pd.concat(cat_rows)[["cot", "gia_tri", "so_luong", "ti_le_TARGET1"]]
cat_table["ti_le_TARGET1"] = (cat_table["ti_le_TARGET1"] * 100).round(2)
save_table(cat_table.set_index("cot"), "08_default_rate_by_category")

bar_cols = [c for c in ("NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
                        "NAME_FAMILY_STATUS", "OCCUPATION_TYPE") if c in df.columns]
overall = df[TARGET].mean() * 100
fig, axes = plt.subplots(2, 3, figsize=(20, 9))
for ax, col in zip(axes.ravel(), bar_cols):
    t = df.groupby(col)[TARGET].mean().mul(100).sort_values()
    sns.barplot(x=t.values, y=t.index, color="#C44E52", ax=ax)
    ax.axvline(overall, color="black", ls="--", lw=1)
    ax.set_title(f"Tỷ lệ TARGET=1 theo {col}", fontsize=10)
    ax.set_xlabel("%")
    ax.set_ylabel("")
for ax in axes.ravel()[len(bar_cols):]:
    ax.axis("off")
fig.suptitle(f"Tỷ lệ có khó khăn thanh toán theo nhóm (đường nét đứt = trung bình {overall:.2f}%)", y=1.0)
save_fig("08_default_rate_by_category")

# %% [markdown]
# ## 9. Tóm tắt nhanh để đưa vào báo cáo
# %%
summary = [
    f"- Kích thước: {df.shape[0]:,} dòng x {df.shape[1]} cột ({mem_mb:.0f} MB RAM).",
    f"- Kiểu dữ liệu: {len(cont_num)} số liên tục, {len(binary_num)} cờ nhị phân, {len(cat_cols)} categorical.",
    f"- TARGET: {pct[1]}% là lớp 1 (mất cân bằng {imbalance:.1f}:1) -> dùng ROC-AUC/PR-AUC + class_weight, tránh accuracy.",
    f"- Missing: {len(miss)} cột có thiếu, {(miss['ti_le_thieu_%'] > 60).sum()} cột thiếu > 60%.",
    f"- DAYS_EMPLOYED có giá trị bất thường 365243 ở {anom:,} dòng ({anom / len(df) * 100:.1f}%).",
    f"- Đặc trưng tương quan mạnh nhất với TARGET: {', '.join(corr_rank.head(5).index)}.",
]
(REPORT_DIR / "eda_summary.md").write_text("# Tóm tắt EDA\n\n" + "\n".join(summary) + "\n", encoding="utf-8")
print("\n".join(summary))
print(f"\nHoàn tất. Bảng: {REPORT_DIR}  |  Hình: {FIG_DIR}")
