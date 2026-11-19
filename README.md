# Home Credit Default Risk - EDA, tiền xử lý, Feature Engineering, mô hình hóa và giải thích mô hình

Dataset: https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv

## Cách chạy

```bash
pip install -r requirements.txt
# Tải application_train.csv từ Kaggle, đặt vào data/raw/

# Giai đoạn 1: dữ liệu
python src/01_eda.py            # EDA  -> reports/*.csv, reports/figures/*.png, reports/eda_summary.md
python src/02_preprocess.py     # xử lý + FE -> data/processed/*, models/preprocessor.joblib
python src/03_baseline.py       # Logistic Regression + Random Forest -> reports/baseline_*

# Giai đoạn 2: Gradient Boosting, tối ưu, so sánh, giải thích
python src/04_gbm_baseline.py   # XGBoost + LightGBM mặc định -> reports/gbm_baseline_*
python src/05_imbalance.py      # class weight, undersample, SMOTE -> reports/imbalance_results.csv
python src/06_tune.py           # Optuna tuning (mặc định 30 trial/mô hình) -> reports/tuning_*
python src/07_compare.py        # bảng so sánh các thuật toán + chọn mô hình tốt nhất
python src/08_shap.py           # SHAP cho mô hình tốt nhất -> reports/figures/15-18_*, reports/shap_*
```

Tùy chọn:

```bash
python src/06_tune.py --trials 5 --frac 0.3          # chạy thử nhanh (30% train, 5 trial)
python src/06_tune.py --model lightgbm --trials 50   # chỉ tuning LightGBM
python src/06_tune.py --metric pr_auc --timeout 1800 # tối ưu PR-AUC, tối đa 30 phút mỗi mô hình
python src/07_compare.py --select pr_auc             # chọn mô hình tốt nhất theo PR-AUC (mặc định ROC-AUC)
python src/08_shap.py --n 10000                      # số mẫu valid dùng để tính SHAP (mặc định 5000)
```

## Cấu trúc

```
src/config.py            đường dẫn, seed, ngưỡng xử lý (sửa tham số ở đây)
src/features.py          clean_raw() + add_features()  (không fit, không rò rỉ dữ liệu)
src/preprocessing.py     HomeCreditPreprocessor: missing, ngoại lệ, encoding, scaling (fit trên train)
src/modeling_utils.py    hàm dùng chung cho 04-08: đọc dữ liệu, evaluate(), tạo mô hình, early stopping
src/01_eda.py ... 03_baseline.py    EDA, tiền xử lý, baseline
src/04_gbm_baseline.py   XGBoost, LightGBM tham số mặc định
src/05_imbalance.py      thử nghiệm xử lý mất cân bằng
src/06_tune.py           tối ưu tham số bằng Optuna
src/07_compare.py        bảng so sánh, ROC/PR, chọn mô hình tốt nhất
src/08_shap.py           giải thích mô hình bằng SHAP
data/processed/          dữ liệu đã xử lý cho cả nhóm dùng
reports/                 bảng + hình EDA, kết quả baseline, tuning, so sánh, SHAP
models/                  preprocessor.joblib, các mô hình đã huấn luyện, best_model.joblib
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

## Mô hình hóa (04-08)

### Quy trình đánh giá

- Mô hình cây dùng `train_processed` / `valid_processed`.
- Mỗi mô hình được huấn luyện trên 90% train, 10% train còn lại dùng cho **early stopping** (100 vòng). Tập **valid không dùng để chọn tham số hay early stopping**, chỉ dùng để đánh giá cuối.
- Chỉ số: ROC-AUC, PR-AUC (chỉ số chính), Precision, Recall, F1 ở ngưỡng 0.5 và ở ngưỡng tối ưu F1 (`@best`). Ngưỡng `@best` chọn trên valid nên chỉ mang tính tham khảo.
- Cách tính chỉ số trùng với `03_baseline.py` nên các bảng so sánh được với nhau.

### 04 - XGBoost và LightGBM

Hai mô hình với tham số mặc định hợp lý và `scale_pos_weight` bằng tỷ lệ âm/dương (~11.4). Xuất `reports/gbm_baseline_results.csv`, `figures/11_gbm_feature_importance.png`.

### 05 - Class weight và xử lý mất cân bằng

Chỉ áp dụng lên phần train, tập early stopping và valid giữ phân phối gốc.

| Chiến lược | Mô tả |
|---|---|
| `none` | Không xử lý (`scale_pos_weight = 1`) |
| `sqrt_weight` | `scale_pos_weight = sqrt(tỷ lệ âm/dương)` |
| `class_weight` | `scale_pos_weight = tỷ lệ âm/dương` |
| `undersample` | Giữ toàn bộ lớp 1, lấy ngẫu nhiên lớp 0 theo tỷ lệ 3:1 |
| `smote` | SMOTE nâng lớp 1 lên 30% lớp 0 (cần `imbalanced-learn`, thiếu thì tự bỏ qua) |

Xuất `reports/imbalance_results.csv`, `figures/12_imbalance_comparison.png`.

Lưu ý: ROC-AUC và PR-AUC chỉ đo khả năng xếp hạng nên ít thay đổi theo class weight. Precision/Recall/F1 ở ngưỡng 0.5 mới thay đổi rõ (xem các cột `*@0.5`). SMOTE nội suy trên cờ 0/1 và one-hot nên sinh giá trị lẻ cho các cột này.

### 06 - Tối ưu tham số (Optuna)

- Stratified K-fold (mặc định 3 fold) trên train, tối ưu ROC-AUC (hoặc `--metric pr_auc`).
- Không gian tìm kiếm có cả `scale_pos_weight` (`none` / `sqrt` / `full`).
- Xuất `reports/tuning_<model>_best.json` và `reports/tuning_<model>_trials.csv`.
- Đây là bước chạy lâu nhất, nên thử nhanh bằng `--trials 5 --frac 0.3` trước.

### 07 - Bảng so sánh các thuật toán

Huấn luyện lại XGBoost và LightGBM với tham số đã tuning, gộp với Logistic Regression, Random Forest và bản mặc định. Xuất:

- `reports/model_comparison.csv`, `reports/model_comparison.md`: bảng so sánh tất cả mô hình
- `reports/figures/13_model_roc_pr.png`, `14_model_metrics_bar.png`
- `models/best_model.joblib`, `models/best_model.json`: mô hình tốt nhất (theo ROC-AUC, đổi bằng `--select`)
- `reports/all_valid_predictions.csv`: xác suất dự đoán trên valid của mọi mô hình

### 08 - Giải thích bằng SHAP

SHAP (`TreeExplainer`) trên mẫu ngẫu nhiên của valid. Xuất:

- `figures/15_shap_summary.png` (beeswarm), `16_shap_bar.png` (mean |SHAP|)
- `figures/17_shap_dependence_<feature>.png`: 4 đặc trưng quan trọng nhất
- `figures/18_shap_waterfall_high_risk.png`, `18_shap_waterfall_low_risk.png`: giải thích từng khách hàng
- `reports/shap_importance.csv`: mean |SHAP|, hướng ảnh hưởng, có phải feature tự tạo không
- `reports/shap_analysis.md`: bảng top đặc trưng tự động, dùng làm khung cho phần phân tích trong báo cáo

Hướng ảnh hưởng tính bằng tương quan hạng (Spearman) giữa giá trị đặc trưng và SHAP value. Đây là quan hệ mô hình học được, không phải quan hệ nhân quả.

## Trong đánh giá

TARGET mất cân bằng (~8% là lớp 1) nên dùng **ROC-AUC và PR-AUC**, không dùng accuracy. Cả hai baseline dùng `class_weight` cân bằng.

## Gợi ý `.gitignore`

```
__pycache__/
data/raw/
data/processed/
models/*.joblib
```