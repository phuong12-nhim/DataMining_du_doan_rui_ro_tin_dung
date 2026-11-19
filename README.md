# Dự đoán rủi ro tín dụng - Home Credit Default Risk

Bài tập lớn môn **Khai phá dữ liệu**: dự đoán khả năng khách hàng **không trả được nợ** (`TARGET = 1`) trên bộ dữ liệu Home Credit.

Quy trình đầy đủ: EDA → tiền xử lý + Feature Engineering → mô hình baseline → xử lý mất cân bằng lớp → tinh chỉnh siêu tham số → so sánh mô hình → giải thích mô hình bằng SHAP.

- **Dữ liệu:** [application_train.csv](https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv) (Kaggle, Home Credit Default Risk)
- **Bài toán:** phân loại nhị phân, lớp 1 (vỡ nợ) chỉ chiếm khoảng 8%
- **Độ đo chính:** ROC-AUC và PR-AUC (không dùng accuracy vì dữ liệu mất cân bằng)

## Quy trình tổng quan

| Bước | Script | Nội dung | Kết quả chính |
|---|---|---|---|
| 1 | `01_eda.py` | Khám phá dữ liệu | `reports/*.csv`, `reports/figures/*.png`, `reports/eda_summary.md` |
| 2 | `02_preprocess.py` | Làm sạch, Feature Engineering, mã hóa, chuẩn hóa, chia train/valid | `data/processed/*`, `models/preprocessor.joblib` |
| 3 | `03_baseline.py` | Logistic Regression, Random Forest | `reports/baseline_*`, `models/logreg_baseline.joblib`, `models/rf_baseline.joblib` |
| 4 | `04_gbm_baseline.py` | XGBoost, LightGBM với tham số mặc định | `reports/gbm_baseline_*`, `models/*_default.joblib` |
| 5 | `05_imbalance.py` | So sánh các chiến lược xử lý mất cân bằng | `reports/imbalance_results.csv`, hình `12_*` |
| 6 | `06_tune.py` | Tinh chỉnh siêu tham số bằng Optuna (CV phân tầng) | `reports/tuning_*_best.json`, `reports/tuning_*_trials.csv` |
| 7 | `07_compare.py` | So sánh toàn bộ mô hình, chọn mô hình tốt nhất | `reports/model_comparison.{csv,md}`, `models/best_model.joblib`, hình `13_*`, `14_*` |
| 8 | `08_shap.py` | Giải thích mô hình tốt nhất bằng SHAP | `reports/shap_importance.csv`, `reports/shap_analysis.md`, hình `15_*` đến `18_*` |

## Cách chạy

```bash
pip install -r requirements.txt
pip install xgboost lightgbm optuna shap imbalanced-learn   # cho bước 4 đến 8
# Tải application_train.csv từ Kaggle, đặt vào data/raw/

python src/01_eda.py
python src/02_preprocess.py
python src/03_baseline.py
python src/04_gbm_baseline.py
python src/05_imbalance.py
python src/06_tune.py --model both --trials 30 --folds 3 --metric roc_auc
python src/07_compare.py --select roc_auc
python src/08_shap.py --n 5000 --top 20
```

Các script phải chạy **theo thứ tự**, vì bước sau đọc kết quả của bước trước.

Tùy chọn của một số script:

| Script | Tham số | Ý nghĩa |
|---|---|---|
| `06_tune.py` | `--model {xgboost,lightgbm,both}` | Mô hình cần tinh chỉnh (mặc định `both`) |
| | `--trials`, `--folds` | Số lần thử Optuna (mặc định 30) và số fold CV (mặc định 3) |
| | `--metric {roc_auc,pr_auc}` | Độ đo tối ưu hóa |
| | `--frac`, `--timeout` | Dùng một phần dữ liệu train / giới hạn thời gian (giây) để chạy nhanh hơn |
| `07_compare.py` | `--select {roc_auc,pr_auc}` | Tiêu chí chọn mô hình tốt nhất |
| `08_shap.py` | `--n`, `--top` | Số mẫu tính SHAP (mặc định 5000), số đặc trưng hiển thị (mặc định 20) |

Mỗi file `.py` dùng dấu `# %%` nên mở như notebook được trong VS Code (hoặc `jupytext --to ipynb`).

## Cấu trúc thư mục

```
src/
  config.py            đường dẫn, seed, ngưỡng xử lý (sửa tham số ở đây)
  features.py          clean_raw() + add_features()  (không fit, không rò rỉ dữ liệu)
  preprocessing.py     HomeCreditPreprocessor: missing, ngoại lệ, encoding, scaling (fit trên train)
  modeling_utils.py    hàm dùng chung cho bước 4 đến 8: load_train_valid, train_with_es, evaluate, ...
  01_eda.py ... 08_shap.py
data/
  raw/                 application_train.csv (tải từ Kaggle)
  processed/           dữ liệu đã xử lý cho cả nhóm dùng
reports/               bảng kết quả (.csv, .json, .md)
reports/figures/       hình EDA, ROC/PR, so sánh mô hình, SHAP
models/                preprocessor, các mô hình baseline / default / tuned, best_model
```

## Dữ liệu đã xử lý (dành cho các thành viên)

| File | Dùng cho |
|---|---|
| `train_processed`, `valid_processed` | Cây/ensemble: Random Forest, XGBoost, LightGBM (đã impute + encode, chưa scale) |
| `train_scaled`, `valid_scaled` | Mô hình tuyến tính, SVM, NN (biến liên tục đã chuẩn hóa) |
| `feature_dictionary.csv` | Mô tả từng cột: loại xử lý, cột gốc, có phải feature tự tạo không |

- Mỗi file có `SK_ID_CURR` và `TARGET` ở hai cột đầu, còn lại là đặc trưng, không còn NaN.
- Chia train/valid 80/20, stratify theo `TARGET`, `random_state=42`. **Mọi người dùng chung cách chia này** để kết quả so sánh được.
- Đọc dữ liệu: `pd.read_parquet("data/processed/train_processed.parquet")` (hoặc `.csv.gz` nếu không có pyarrow).
- Áp dụng lên `application_test.csv`: `add_features(clean_raw(df))` rồi `joblib.load("models/preprocessor.joblib").transform(...)`.

## Tiền xử lý và Feature Engineering

**Quyết định xử lý chính**

- **DAYS_EMPLOYED = 365243** (người hưu/không đi làm) → NaN + cờ `DAYS_EMPLOYED_ANOM`.
- **CODE_GENDER = XNA, ORGANIZATION_TYPE = XNA** → coi là thiếu.
- **Missing**: bỏ cột thiếu > 60%; cột thiếu >= 5% thêm cờ `*_ISNULL`; số → median, cờ → mode, categorical → nhóm `Missing`.
- **Cột trùng lặp**: nhóm nhà ở `*_MEDI`, `*_MODE` gần giống `*_AVG` → chỉ giữ `*_AVG`. Cột hằng số bị bỏ.
- **Ngoại lệ**: cắt ở phân vị 0.1% và 99.9% (học từ train), không xóa dòng.
- **Encoding**: categorical 2 mức → 0/1; nhiều mức → one-hot, nhóm < 1% gom `Other`.
- **Scaling**: StandardScaler chỉ cho biến liên tục, chỉ có trong file `*_scaled`.

**Feature Engineering (~35 đặc trưng, xem `src/features.py`)**

- Tỷ lệ tài chính: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `CREDIT_GOODS_RATIO`
- Thời gian: `AGE_YEARS`, `EMPLOYED_YEARS`, `EMPLOYED_TO_AGE_RATIO`, `ID_PUBLISH_YEARS`, ...
- EXT_SOURCE: mean / min / max / std / prod / weighted / số lần thiếu
- Hộ gia đình: `INCOME_PER_PERSON`, `CREDIT_PER_PERSON`, `CHILDREN_RATIO`
- Tổng hợp: `DOCUMENT_COUNT`, `CONTACT_COUNT`, `ADDRESS_MISMATCH_COUNT`, `SOCIAL_DEF_RATE_30/60`, `BUREAU_ENQUIRY_TOTAL`

## Mô hình hóa

**Các mô hình** (đánh giá trên cùng tập valid):

| Nhóm | Mô hình |
|---|---|
| Baseline | Logistic Regression, Random Forest (`class_weight` cân bằng) |
| Gradient boosting | XGBoost, LightGBM (mặc định, `scale_pos_weight` = tỷ lệ âm/dương) |
| Tinh chỉnh | XGBoost (tuned), LightGBM (tuned) |

Các mô hình boosting dùng early stopping trên một phần tách riêng từ tập train (valid không bị dùng để dừng sớm).

**Xử lý mất cân bằng lớp** (`05_imbalance.py`), thử trên cả XGBoost và LightGBM:

| Chiến lược | Mô tả |
|---|---|
| `none` | không xử lý (`scale_pos_weight = 1`) |
| `sqrt_weight` | trọng số lớp dương = căn bậc hai của tỷ lệ âm/dương |
| `class_weight` | trọng số lớp dương = tỷ lệ âm/dương |
| `undersample` | giữ toàn bộ lớp dương, lấy ngẫu nhiên 3 mẫu âm cho mỗi mẫu dương |
| `smote` | SMOTE với `sampling_strategy = 0.3` (cần `imbalanced-learn`) |

**Tinh chỉnh siêu tham số** (`06_tune.py`): Optuna (TPE sampler, MedianPruner), StratifiedKFold; không gian tìm kiếm gồm các tham số cấu trúc cây, lấy mẫu, regularization và chế độ `scale_pos_weight` (`none` / `sqrt` / `full`).

**Chọn mô hình tốt nhất** (`07_compare.py`): chọn trong các bản đã tuning theo ROC-AUC (hoặc PR-AUC với `--select pr_auc`), lưu vào `models/best_model.joblib` kèm thông tin ở `models/best_model.json`.

## Kết quả

Số liệu chi tiết nằm trong:

- `reports/model_comparison.md` / `.csv`: bảng so sánh toàn bộ mô hình (ROC-AUC, PR-AUC, Precision / Recall / F1 ở ngưỡng 0.5 và ngưỡng tối ưu F1)
- `reports/imbalance_results.csv`: so sánh các chiến lược mất cân bằng
- `reports/figures/13_model_roc_pr.png`, `14_model_metrics_bar.png`: đường cong ROC/PR và so sánh các độ đo

<!-- TODO: điền bảng kết quả thật sau khi chạy xong, ví dụ:
| Mô hình | ROC-AUC | PR-AUC | F1@best |
|---|---|---|---|
| Logistic Regression | ... | ... | ... |
| Random Forest | ... | ... | ... |
| XGBoost (tuned) | ... | ... | ... |
| LightGBM (tuned) | ... | ... | ... |
-->

## Giải thích mô hình (SHAP)

`08_shap.py` tính SHAP trên mô hình tốt nhất và tạo:

- Mức độ quan trọng toàn cục (mean |SHAP|) kèm **hướng ảnh hưởng** của từng đặc trưng (tăng hay giảm rủi ro), đánh dấu đặc trưng nào là feature tự tạo → `reports/shap_importance.csv`, `reports/shap_analysis.md`
- Biểu đồ summary, bar, dependence (4 đặc trưng đầu) → `reports/figures/15_*`, `16_*`, `17_*`
- Waterfall cho 1 khách hàng rủi ro cao và 1 khách hàng rủi ro thấp → `reports/figures/18_*`

Lưu ý: SHAP phản ánh quan hệ mô hình học được, không phải quan hệ nhân quả.

## Lưu ý đánh giá

- `TARGET` mất cân bằng (~8% là lớp 1) nên dùng **ROC-AUC và PR-AUC**, không dùng accuracy.
- ROC-AUC / PR-AUC chỉ đo khả năng xếp hạng nên ít đổi theo class weight; Precision / Recall / F1 ở ngưỡng 0.5 mới thay đổi rõ giữa các chiến lược.
- Ngưỡng tối ưu F1 (`@best`) được chọn trên tập valid nên chỉ mang tính tham khảo.
