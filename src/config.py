"""Cấu hình chung cho toàn nhóm. Mọi script đều import từ đây để đường dẫn và tham số thống nhất."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"

RAW_FILE = RAW_DIR / "application_train.csv"

for d in (RAW_DIR, PROCESSED_DIR, REPORT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---- Tham số tiền xử lý ----
MISSING_DROP_THRESHOLD = 0.60     # bỏ cột thiếu > 60%
MISSING_FLAG_THRESHOLD = 0.05     # cột thiếu >= 5% thì thêm cờ *_ISNULL
RARE_CATEGORY_MIN_FREQ = 0.01     # nhóm hiếm (<1%) gom thành "Other"
WINSOR_QUANTILES = (0.001, 0.999) # cắt ngoại lệ ở 0.1% và 99.9% (fit trên train)
DAYS_EMPLOYED_ANOMALY = 365243    # giá trị bất thường nổi tiếng của DAYS_EMPLOYED
