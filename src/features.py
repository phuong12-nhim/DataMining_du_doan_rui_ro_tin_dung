"""Feature Engineering từ các cột có sẵn trong application_train.

Đây là các hàm "thuần" (không fit tham số từ dữ liệu) nên có thể áp dụng cho train, valid, test
mà không gây rò rỉ dữ liệu (data leakage). Cột nào không tồn tại sẽ được bỏ qua.
"""
import numpy as np
import pandas as pd

from config import DAYS_EMPLOYED_ANOMALY


def _safe_div(a, b):
    """Chia an toàn: mẫu số 0 hoặc NaN -> NaN."""
    b = b.replace(0, np.nan)
    return a / b


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch các giá trị bất thường đã biết của dataset Home Credit."""
    df = df.copy()

    # DAYS_EMPLOYED = 365243 (~1000 năm) là mã cho người nghỉ hưu/không đi làm -> NaN + cờ
    if "DAYS_EMPLOYED" in df:
        anom = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
        df["DAYS_EMPLOYED_ANOM"] = anom.astype(int)
        df.loc[anom, "DAYS_EMPLOYED"] = np.nan

    # CODE_GENDER có 4 dòng 'XNA' -> NaN (sẽ được impute sau)
    if "CODE_GENDER" in df:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # Các cột có giá trị 'XNA' hoặc 'Unknown' thực chất là thiếu thông tin
    for c in ("ORGANIZATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE"):
        if c in df:
            df[c] = df[c].replace({"XNA": np.nan, "Unknown": np.nan})

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng mới. Trả về DataFrame mới (không sửa df gốc)."""
    df = df.copy()
    has = lambda *cols: all(c in df.columns for c in cols)  # noqa: E731
    new = {}

    # ---- 1. Tuổi và thời gian (các cột DAYS_* là số âm, tính từ ngày nộp đơn) ----
    if has("DAYS_BIRTH"):
        new["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if has("DAYS_EMPLOYED"):
        new["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    if has("DAYS_EMPLOYED", "DAYS_BIRTH"):
        new["EMPLOYED_TO_AGE_RATIO"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    if has("DAYS_REGISTRATION"):
        new["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    if has("DAYS_ID_PUBLISH"):
        new["ID_PUBLISH_YEARS"] = -df["DAYS_ID_PUBLISH"] / 365.25
    if has("DAYS_LAST_PHONE_CHANGE"):
        new["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25

    # ---- 2. Tỷ lệ tài chính (mạnh nhất trong bài toán này) ----
    if has("AMT_CREDIT", "AMT_INCOME_TOTAL"):
        new["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    if has("AMT_ANNUITY", "AMT_INCOME_TOTAL"):
        new["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    if has("AMT_CREDIT", "AMT_ANNUITY"):
        # kỳ hạn vay xấp xỉ (số kỳ trả góp)
        new["CREDIT_TERM"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    if has("AMT_CREDIT", "AMT_GOODS_PRICE"):
        new["CREDIT_GOODS_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
        new["GOODS_CREDIT_DIFF"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    if has("AMT_INCOME_TOTAL"):
        new["LOG_INCOME"] = np.log1p(df["AMT_INCOME_TOTAL"].clip(lower=0))
    if has("AMT_CREDIT"):
        new["LOG_CREDIT"] = np.log1p(df["AMT_CREDIT"].clip(lower=0))

    # ---- 3. Hộ gia đình ----
    if has("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"):
        new["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    if has("AMT_CREDIT", "CNT_FAM_MEMBERS"):
        new["CREDIT_PER_PERSON"] = _safe_div(df["AMT_CREDIT"], df["CNT_FAM_MEMBERS"])
    if has("CNT_CHILDREN", "CNT_FAM_MEMBERS"):
        new["CHILDREN_RATIO"] = _safe_div(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
    if has("AMT_INCOME_TOTAL", "CNT_CHILDREN"):
        new["INCOME_PER_CHILD"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_CHILDREN"] + 1)

    # ---- 4. Tài sản ----
    if has("OWN_CAR_AGE", "DAYS_BIRTH"):
        new["CAR_AGE_TO_AGE_RATIO"] = _safe_div(df["OWN_CAR_AGE"], -df["DAYS_BIRTH"] / 365.25)
    if has("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        new["OWN_CAR_AND_REALTY"] = (
            (df["FLAG_OWN_CAR"] == "Y") & (df["FLAG_OWN_REALTY"] == "Y")
        ).astype(int)

    # ---- 5. EXT_SOURCE_1/2/3 (điểm tín nhiệm từ nguồn ngoài - đặc trưng mạnh nhất) ----
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
    if ext:
        e = df[ext]
        new["EXT_SOURCE_MEAN"] = e.mean(axis=1)
        new["EXT_SOURCE_MIN"] = e.min(axis=1)
        new["EXT_SOURCE_MAX"] = e.max(axis=1)
        new["EXT_SOURCE_STD"] = e.std(axis=1)
        # Trung bình nhân của các điểm CÓ GIÁ TRỊ (thay cho tích skipna=False cũ: tích bị NaN khi thiếu bất kỳ
        # điểm nào -> thiếu > 60% và bị loại ở bước tiền xử lý). Trung bình nhân so sánh được dù số điểm khác nhau.
        new["EXT_SOURCE_GEOMEAN"] = np.exp(np.log(e.clip(lower=1e-6)).mean(axis=1))
        new["EXT_SOURCE_NULL_COUNT"] = e.isna().sum(axis=1)
    # Trung bình có trọng số (2:3:4) CHUẨN HÓA theo các điểm có giá trị -> chỉ NaN khi thiếu cả 3 điểm
    w_ext = {"EXT_SOURCE_1": 2, "EXT_SOURCE_2": 3, "EXT_SOURCE_3": 4}
    avail = [c for c in w_ext if c in df.columns]
    if avail:
        wts = pd.Series({c: w_ext[c] for c in avail})
        new["EXT_SOURCE_WEIGHTED"] = _safe_div((df[avail] * wts).sum(axis=1, min_count=1),
                                               df[avail].notna().mul(wts).sum(axis=1))
    if has("EXT_SOURCE_3", "DAYS_BIRTH"):
        new["EXT3_X_AGE"] = df["EXT_SOURCE_3"] * (-df["DAYS_BIRTH"] / 365.25)

    # ---- 6. Tài liệu / liên lạc ----
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        new["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    contact_cols = [c for c in ("FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                                "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL") if c in df.columns]
    if contact_cols:
        new["CONTACT_COUNT"] = df[contact_cols].sum(axis=1)

    # ---- 7. Vùng / khu vực ----
    if has("REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"):
        new["REGION_RATING_MEAN"] = df[["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]].mean(axis=1)
    addr_flags = [c for c in df.columns if c.startswith(("REG_REGION_NOT", "REG_CITY_NOT", "LIVE_REGION_NOT", "LIVE_CITY_NOT"))]
    if addr_flags:
        new["ADDRESS_MISMATCH_COUNT"] = df[addr_flags].sum(axis=1)

    # ---- 8. Vòng quan hệ xã hội & truy vấn tín dụng ----
    if has("DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"):
        new["SOCIAL_DEF_RATE_30"] = _safe_div(df["DEF_30_CNT_SOCIAL_CIRCLE"], df["OBS_30_CNT_SOCIAL_CIRCLE"])
    if has("DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"):
        new["SOCIAL_DEF_RATE_60"] = _safe_div(df["DEF_60_CNT_SOCIAL_CIRCLE"], df["OBS_60_CNT_SOCIAL_CIRCLE"])
    enq = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if enq:
        new["BUREAU_ENQUIRY_TOTAL"] = df[enq].sum(axis=1, min_count=1)

    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
