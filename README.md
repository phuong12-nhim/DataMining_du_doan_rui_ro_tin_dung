# Dự đoán rủi ro tín dụng - Home Credit Default Risk

Bài tập lớn môn **Khai phá dữ liệu**: dự đoán khả năng khách hàng **không trả được nợ** (`TARGET = 1`) trên bộ dữ liệu Home Credit.

Quy trình đầy đủ: EDA → tiền xử lý + Feature Engineering → mô hình baseline → mô hình Gradient Boosting → xử lý mất cân bằng lớp → tinh chỉnh siêu tham số → so sánh mô hình → giải thích mô hình bằng SHAP.

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
# Tải application_train.csv từ Kaggle, đặt vào data/raw/

# Giai đoạn 1: dữ liệu
python src/01_eda.py            # EDA
python src/02_preprocess.py     # xử lý + FE
python src/03_baseline.py       # Logistic Regression + Random Forest

# Giai đoạn 2: Gradient Boosting, tối ưu, so sánh, giải thích
python src/04_gbm_baseline.py   # XGBoost + LightGBM mặc định
python src/05_imbalance.py      # class weight, undersample, SMOTE
python src/06_tune.py           # Optuna tuning (mặc định 30 trial/mô hình)
python src/07_compare.py        # bảng so sánh + chọn mô hình tốt nhất
python src/08_shap.py           # SHAP cho mô hình tốt nhất
```

Các script phải chạy **theo thứ tự**, vì bước sau đọc kết quả của bước trước. Mỗi file `.py` dùng dấu `# %%` nên mở như notebook được trong VS Code (hoặc `jupytext --to ipynb`).

**Tùy chọn của một số script**

| Script | Tham số | Ý nghĩa |
|---|---|---|
| `06_tune.py` | `--model {xgboost,lightgbm,both}` | Mô hình cần tinh chỉnh (mặc định `both`) |
| | `--trials`, `--folds` | Số lần thử Optuna (mặc định 30) và số fold CV (mặc định 3) |
| | `--metric {roc_auc,pr_auc}` | Độ đo tối ưu hóa (mặc định `roc_auc`) |
| | `--frac`, `--timeout` | Dùng một phần dữ liệu train / giới hạn thời gian (giây) để chạy nhanh hơn |
| `07_compare.py` | `--select {roc_auc,pr_auc}` | Tiêu chí chọn mô hình tốt nhất (mặc định `roc_auc`) |
| `08_shap.py` | `--n`, `--top` | Số mẫu tính SHAP (mặc định 5000), số đặc trưng hiển thị (mặc định 20) |

Ví dụ:

```bash
python src/06_tune.py --trials 5 --frac 0.3          # chạy thử nhanh (30% train, 5 trial)
python src/06_tune.py --model lightgbm --trials 50   # chỉ tuning LightGBM
python src/06_tune.py --metric pr_auc --timeout 1800 # tối ưu PR-AUC, tối đa 30 phút mỗi mô hình
python src/07_compare.py --select pr_auc
python src/08_shap.py --n 10000
```

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
| Gradient boosting | XGBoost, LightGBM (mặc định, `scale_pos_weight` = tỷ lệ âm/dương ≈ 11.4) |
| Tinh chỉnh | XGBoost (tuned), LightGBM (tuned) |

**Quy trình đánh giá**

- Mô hình cây dùng `train_processed` / `valid_processed`.
- Mỗi mô hình boosting được huấn luyện trên 90% train, 10% train còn lại dùng cho **early stopping** (100 vòng). Tập **valid không dùng để chọn tham số hay dừng sớm**, chỉ dùng để đánh giá cuối.
- Chỉ số: ROC-AUC, PR-AUC (chỉ số chính), Precision, Recall, F1 ở ngưỡng 0.5 và ở ngưỡng tối ưu F1 (`@best`).
- Cách tính chỉ số trùng với `03_baseline.py` nên các bảng so sánh được với nhau.

### Bước 4: XGBoost và LightGBM

Hai mô hình với tham số mặc định hợp lý và `scale_pos_weight` bằng tỷ lệ âm/dương. Xuất `reports/gbm_baseline_results.csv`, `figures/11_gbm_feature_importance.png`.

### Bước 5: Xử lý mất cân bằng lớp

Thử trên cả XGBoost và LightGBM. Chỉ áp dụng lên phần train, tập early stopping và valid giữ phân phối gốc.

| Chiến lược | Mô tả |
|---|---|
| `none` | Không xử lý (`scale_pos_weight = 1`) |
| `sqrt_weight` | Trọng số lớp dương = căn bậc hai của tỷ lệ âm/dương |
| `class_weight` | Trọng số lớp dương = tỷ lệ âm/dương |
| `undersample` | Giữ toàn bộ lớp dương, lấy ngẫu nhiên 3 mẫu âm cho mỗi mẫu dương |
| `smote` | SMOTE với `sampling_strategy = 0.3` (cần `imbalanced-learn`, thiếu thì tự bỏ qua) |

Xuất `reports/imbalance_results.csv`, `figures/12_imbalance_comparison.png`. SMOTE nội suy trên cờ 0/1 và one-hot nên sinh giá trị lẻ cho các cột này.

### Bước 6: Tinh chỉnh siêu tham số

- Optuna (TPE sampler, MedianPruner), StratifiedKFold (mặc định 3 fold) trên train, tối ưu ROC-AUC (hoặc `--metric pr_auc`).
- Không gian tìm kiếm gồm tham số cấu trúc cây, lấy mẫu, regularization và chế độ `scale_pos_weight` (`none` / `sqrt` / `full`).
- Xuất `reports/tuning_<model>_best.json` và `reports/tuning_<model>_trials.csv`.
- Đây là bước chạy lâu nhất, nên thử nhanh bằng `--trials 5 --frac 0.3` trước.

### Bước 7: So sánh và chọn mô hình tốt nhất

Huấn luyện lại XGBoost và LightGBM với tham số đã tuning, gộp với Logistic Regression, Random Forest và bản mặc định. Mô hình tốt nhất được chọn trong các bản đã tuning theo ROC-AUC (hoặc PR-AUC với `--select pr_auc`). Xuất:

- `reports/model_comparison.csv`, `reports/model_comparison.md`: bảng so sánh tất cả mô hình
- `reports/figures/13_model_roc_pr.png`, `reports/figures/14_model_metrics_bar.png`
- `models/best_model.joblib`, `models/best_model.json`: mô hình tốt nhất và thông tin kèm theo
- `reports/all_valid_predictions.csv`: xác suất dự đoán trên valid của mọi mô hình

## Kết quả

Số liệu chi tiết nằm trong:

- `reports/model_comparison.md` / `.csv`: bảng so sánh toàn bộ mô hình (ROC-AUC, PR-AUC, Precision / Recall / F1 ở ngưỡng 0.5 và ngưỡng tối ưu F1)
- `reports/imbalance_results.csv`: so sánh các chiến lược mất cân bằng
- `reports/figures/13_model_roc_pr.png`, `reports/figures/14_model_metrics_bar.png`: đường cong ROC/PR và so sánh các độ đo

<!-- TODO: điền bảng kết quả thật sau khi chạy xong, ví dụ:
| Mô hình | ROC-AUC | PR-AUC | F1@best |
|---|---|---|---|
| Logistic Regression | ... | ... | ... |
| Random Forest | ... | ... | ... |
| XGBoost (tuned) | ... | ... | ... |
| LightGBM (tuned) | ... | ... | ... |
-->

## Giải thích mô hình (SHAP)

`08_shap.py` dùng `TreeExplainer` trên mẫu ngẫu nhiên của valid, với mô hình tốt nhất, và tạo:

- Mức độ quan trọng toàn cục (mean |SHAP|) kèm **hướng ảnh hưởng** của từng đặc trưng (tăng hay giảm rủi ro), đánh dấu đặc trưng nào là feature tự tạo → `reports/shap_importance.csv`, `reports/shap_analysis.md` (bảng top đặc trưng tự động, dùng làm khung cho phần phân tích trong báo cáo)
- Biểu đồ summary (beeswarm), bar, dependence cho 4 đặc trưng đầu → `reports/figures/15_*`, `16_*`, `17_*`
- Waterfall cho 1 khách hàng rủi ro cao và 1 khách hàng rủi ro thấp → `reports/figures/18_*`

Hướng ảnh hưởng tính bằng tương quan hạng (Spearman) giữa giá trị đặc trưng và SHAP value. SHAP phản ánh quan hệ mô hình học được, không phải quan hệ nhân quả.

## Trong đánh giá

- `TARGET` mất cân bằng (~8% là lớp 1) nên dùng **ROC-AUC và PR-AUC**, không dùng accuracy. Cả hai baseline dùng `class_weight` cân bằng.
- ROC-AUC / PR-AUC chỉ đo khả năng xếp hạng nên ít đổi theo class weight; Precision / Recall / F1 ở ngưỡng 0.5 mới thay đổi rõ giữa các chiến lược.
- Ngưỡng tối ưu F1 (`@best`) được chọn trên tập valid nên chỉ mang tính tham khảo.

## Gợi ý `.gitignore`

```
__pycache__/
data/raw/
data/processed/
models/*.joblib
```