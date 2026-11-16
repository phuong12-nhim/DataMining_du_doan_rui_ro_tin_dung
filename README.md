# Home Credit Default Risk - EDA, tiền xử lý, Feature Engineering, baseline

Dataset: https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv

## Cách chạy

```bash
pip install -r requirements.txt
# Tải application_train.csv từ Kaggle, đặt vào data/raw/
python src/01_eda.py          # EDA  -> reports/*.csv, reports/figures/*.png, reports/eda_summary.md
python src/02_preprocess.py   # xử lý + FE -> data/processed/*, models/preprocessor.joblib
python src/03_baseline.py     # Logistic Regression + Random Forest -> reports/baseline_*
```

Mỗi file `.py` dùng dấu `# %%` nên mở như notebook được trong VS Code (hoặc `jupytext --to ipynb`).

## Cấu trúc

```
src/config.py          đường dẫn, seed, ngưỡng xử lý (sửa tham số ở đây)
src/features.py        clean_raw() + add_features()  (không fit, không rò rỉ dữ liệu)
src/preprocessing.py   HomeCreditPreprocessor: missing, ngoại lệ, encoding, scaling (fit trên train)
src/01_eda.py ... 03_baseline.py
data/processed/        dữ liệu đã xử lý cho cả nhóm dùng
reports/               bảng + hình EDA, kết quả baseline
models/                preprocessor.joblib, logreg_baseline.joblib, rf_baseline.joblib
```

## Dữ liệu đã xử lý (dành cho các thành viên khác)

| File | Dùng cho |
|---|---|
| `train_processed`, `valid_processed` | Cây/ensemble: Random Forest, XGBoost, LightGBM (đã impute + encode, chưa scale) |
| `train_scaled`, `valid_scaled` | Mô hình tuyến tính, SVM, NN (biến liên tục đã chuẩn hóa) |
| `feature_dictionary.csv` | Mô tả từng cột: loại xử lý, cột gốc, có phải feature tự tạo không |

- Mỗi file có `SK_ID_CURR` và `TARGET` ở hai cột đầu, còn lại là đặc trưng, không còn NaN.
- Chia train/valid 80/20, stratify theo TARGET, `random_state=42`. **Mọi người dùng chung cách chia này** để kết quả so sánh được.
- Đọc dữ liệu: `pd.read_parquet("data/processed/train_processed.parquet")` (hoặc `.csv.gz` nếu không có pyarrow).
- Áp dụng lên `application_test.csv`: `add_features(clean_raw(df))` rồi `joblib.load("models/preprocessor.joblib").transform(...)`.

## Quyết định xử lý chính

- **DAYS_EMPLOYED = 365243** (người hưu/không đi làm) -> NaN + cờ `DAYS_EMPLOYED_ANOM`.
- **CODE_GENDER = XNA, ORGANIZATION_TYPE = XNA** -> coi là thiếu.
- **Missing**: bỏ cột thiếu > 60%; cột thiếu >= 5% thêm cờ `*_ISNULL`; số -> median, cờ -> mode, categorical -> nhóm `Missing`.
- **Cột trùng lặp**: nhóm nhà ở `*_MEDI`, `*_MODE` gần giống `*_AVG` -> chỉ giữ `*_AVG`. Cột hằng số bị bỏ.
- **Ngoại lệ**: cắt ở phân vị 0.1% và 99.9% (học từ train), không xóa dòng.
- **Encoding**: categorical 2 mức -> 0/1; nhiều mức -> one-hot, nhóm < 1% gom `Other`.
- **Scaling**: StandardScaler chỉ cho biến liên tục, chỉ có trong file `*_scaled`.

## Feature Engineering (~35 đặc trưng, xem `src/features.py`)

- Tỷ lệ tài chính: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `CREDIT_GOODS_RATIO`
- Thời gian: `AGE_YEARS`, `EMPLOYED_YEARS`, `EMPLOYED_TO_AGE_RATIO`, `ID_PUBLISH_YEARS`, ...
- EXT_SOURCE: mean / min / max / std / prod / weighted / số lần thiếu
- Hộ gia đình: `INCOME_PER_PERSON`, `CREDIT_PER_PERSON`, `CHILDREN_RATIO`
- Tổng hợp: `DOCUMENT_COUNT`, `CONTACT_COUNT`, `ADDRESS_MISMATCH_COUNT`, `SOCIAL_DEF_RATE_30/60`, `BUREAU_ENQUIRY_TOTAL`

## Lưu ý đánh giá

TARGET mất cân bằng (~8% là lớp 1) nên dùng **ROC-AUC và PR-AUC**, không dùng accuracy. Cả hai baseline dùng `class_weight` cân bằng.
