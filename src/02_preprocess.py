# %% [markdown]
# # 02 - Tiền xử lý + Feature Engineering + lưu dữ liệu đã xử lý
# Luồng: đọc raw -> clean -> feature engineering -> chia train/valid (stratify) -> fit preprocessor trên train
# -> transform -> lưu.
#
# Chạy: `python src/02_preprocess.py`
# Đầu ra (data/processed/):
#   train_processed / valid_processed   : đã impute + encode, CHƯA scale  (dùng cho Random Forest, XGBoost, LightGBM)
#   train_scaled    / valid_scaled      : như trên nhưng biến liên tục đã chuẩn hóa (dùng cho Logistic Regression, NN)
#   preprocessor.joblib                 : để áp dụng đúng quy trình lên application_test.csv
#   feature_dictionary.csv              : mô tả từng cột (loại xử lý, có phải feature tự tạo không)
# Mỗi file có cột SK_ID_CURR và TARGET ở đầu.

# %%
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ID_COL, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE, RAW_FILE, REPORT_DIR,  # noqa: E402
                    TARGET, TEST_SIZE)
from features import add_features, clean_raw  # noqa: E402
from preprocessing import HomeCreditPreprocessor  # noqa: E402

try:
    import pyarrow  # noqa: F401
    EXT = "parquet"
except ImportError:
    EXT = "csv.gz"


def save_df(df, name):
    path = PROCESSED_DIR / f"{name}.{EXT}"
    if EXT == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, compression="gzip")
    print(f"  đã lưu {path.name}: {df.shape[0]:,} dòng x {df.shape[1]} cột")


# %% [markdown]
# ## 1. Đọc + clean + feature engineering
# %%
raw = pd.read_csv(RAW_FILE)
raw_cols = [c for c in raw.columns if c not in (ID_COL, TARGET)]
print("Raw:", raw.shape)

df = add_features(clean_raw(raw))
n_new = df.shape[1] - raw.shape[1]
print(f"Sau clean + FE: {df.shape}  (+{n_new} cột mới)")

# %% [markdown]
# ## 2. Chia train / valid (stratify theo TARGET để giữ nguyên tỷ lệ mất cân bằng)
# Chia TRƯỚC khi fit preprocessor để không rò rỉ thông tin từ valid.
# %%
train_df, valid_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
print(f"train: {len(train_df):,} (TARGET=1: {train_df[TARGET].mean():.4f}) | "
      f"valid: {len(valid_df):,} (TARGET=1: {valid_df[TARGET].mean():.4f})")

feat_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]

# %% [markdown]
# ## 3. Fit preprocessor trên train, transform cả hai tập
# %%
pre = HomeCreditPreprocessor().fit(train_df[feat_cols])
print(f"Bỏ {len(pre.drop_cols_)} cột (thiếu > {pre.missing_drop_threshold:.0%}, trùng lặp nhóm nhà ở, hoặc hằng số)")
print(f"Còn lại: {len(pre.cont_num_)} số liên tục, {len(pre.binary_num_)} cờ nhị phân, "
      f"{len(pre.binary_cat_)} binary-categorical, {len(pre.multi_cat_)} categorical nhiều mức")
print(f"Tổng số đặc trưng đầu ra: {len(pre.feature_names_)}")


def build(frame, scale):
    X = pre.transform(frame[feat_cols], scale=scale)
    head = frame[[ID_COL, TARGET]].reset_index(drop=True)
    return pd.concat([head, X.reset_index(drop=True)], axis=1)


# %% [markdown]
# ## 4. Kiểm tra chất lượng đầu ra rồi lưu
# %%
train_proc, valid_proc = build(train_df, False), build(valid_df, False)
train_scl, valid_scl = build(train_df, True), build(valid_df, True)

for name, d in [("train_processed", train_proc), ("valid_processed", valid_proc)]:
    assert d.drop(columns=[ID_COL, TARGET]).isna().sum().sum() == 0, f"{name} còn NaN"
assert list(train_proc.columns) == list(valid_proc.columns)
const_cols = [c for c in train_proc.columns[2:] if train_proc[c].nunique() <= 1]
print("Cột hằng số (không mang thông tin):", const_cols or "không có")

save_df(train_proc, "train_processed")
save_df(valid_proc, "valid_processed")
save_df(train_scl, "train_scaled")
save_df(valid_scl, "valid_scaled")

joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
fdict = pre.feature_dictionary(raw_cols)
fdict.to_csv(PROCESSED_DIR / "feature_dictionary.csv", index=False, encoding="utf-8-sig")

dropped = pd.DataFrame({"cot_bi_bo": pre.drop_cols_})
dropped.to_csv(REPORT_DIR / "dropped_columns.csv", index=False, encoding="utf-8-sig")

summary = f"""# Tóm tắt tiền xử lý

- Input: {raw.shape[0]:,} dòng x {raw.shape[1]} cột
- Feature engineering: +{n_new} cột (xem `feature_dictionary.csv`, cột `engineered`)
- Chia dữ liệu: train {len(train_df):,} / valid {len(valid_df):,} (stratify theo TARGET, seed {RANDOM_STATE})
- Cột bị bỏ: {len(pre.drop_cols_)} (danh sách trong `reports/dropped_columns.csv`)
- Số đặc trưng sau xử lý: {len(pre.feature_names_)}
  - numeric liên tục: {len(pre.cont_num_)} | cờ missing (`*_ISNULL`): {len(pre.flag_cols_)}
  - cờ nhị phân: {len(pre.binary_num_) + len(pre.binary_cat_)} | one-hot: {int((fdict['kind'] == 'one_hot').sum())}
- Missing: median (số), mode (cờ nhị phân), nhóm "Missing" (categorical)
- Ngoại lệ: cắt (winsorize) ở phân vị {pre.winsor_quantiles[0]:.1%} - {pre.winsor_quantiles[1]:.1%}, học từ train
- Scaling: StandardScaler cho biến liên tục (chỉ trong file `*_scaled`)
"""
(REPORT_DIR / "preprocess_summary.md").write_text(summary, encoding="utf-8")
print("\n" + summary)
