"""Bộ tiền xử lý dùng chung: missing values, ngoại lệ, encoding, scaling.

Nguyên tắc chống rò rỉ dữ liệu: MỌI tham số (median, phân vị cắt, danh sách category, mean/std)
chỉ được học (fit) từ tập TRAIN, sau đó áp dụng (transform) lên valid/test.

Dùng:
    pre = HomeCreditPreprocessor().fit(X_train)
    X_tr = pre.transform(X_train)               # chưa scale (cho Random Forest / cây quyết định)
    X_tr_s = pre.transform(X_train, scale=True)  # đã scale (cho Logistic Regression / SVM / NN)
"""
import numpy as np
import pandas as pd

from config import (MISSING_DROP_THRESHOLD, MISSING_FLAG_THRESHOLD,
                    RARE_CATEGORY_MIN_FREQ, WINSOR_QUANTILES)


class HomeCreditPreprocessor:
    def __init__(self,
                 missing_drop_threshold=MISSING_DROP_THRESHOLD,
                 missing_flag_threshold=MISSING_FLAG_THRESHOLD,
                 rare_min_freq=RARE_CATEGORY_MIN_FREQ,
                 winsor_quantiles=WINSOR_QUANTILES,
                 drop_redundant_housing=True):
        self.missing_drop_threshold = missing_drop_threshold
        self.missing_flag_threshold = missing_flag_threshold
        self.rare_min_freq = rare_min_freq
        self.winsor_quantiles = winsor_quantiles
        self.drop_redundant_housing = drop_redundant_housing

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame):
        X = X.copy()
        miss_rate = X.isna().mean()

        # 1) Cột bỏ đi: thiếu quá nhiều
        self.drop_cols_ = miss_rate[miss_rate > self.missing_drop_threshold].index.tolist()

        # 2) Cột nhà ở *_MEDI / *_MODE gần như trùng hoàn toàn với *_AVG -> giữ *_AVG
        if self.drop_redundant_housing:
            for c in X.columns:
                for suf in ("_MEDI", "_MODE"):
                    if c.endswith(suf) and c[: -len(suf)] + "_AVG" in X.columns:
                        self.drop_cols_.append(c)
        # 3) Cột hằng số (chỉ 1 giá trị) không mang thông tin
        self.drop_cols_ += [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        self.drop_cols_ = sorted(set(self.drop_cols_))
        X = X.drop(columns=self.drop_cols_)

        # 4) Phân nhóm cột
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        num_cols = [c for c in X.columns if c not in cat_cols]
        self.binary_cat_ = [c for c in cat_cols if X[c].dropna().nunique() <= 2]
        self.multi_cat_ = [c for c in cat_cols if c not in self.binary_cat_]
        self.binary_num_ = [c for c in num_cols if X[c].dropna().nunique() <= 2]
        self.cont_num_ = [c for c in num_cols if c not in self.binary_num_]

        # 5) Binary categorical: ánh xạ 0/1 theo thứ tự alphabet, NaN -> giá trị phổ biến nhất
        self.binary_map_, self.binary_cat_mode_ = {}, {}
        for c in self.binary_cat_:
            levels = sorted(X[c].dropna().unique())
            self.binary_map_[c] = {v: i for i, v in enumerate(levels)}
            self.binary_cat_mode_[c] = self.binary_map_[c][X[c].mode().iloc[0]]

        # 6) Categorical nhiều mức: giữ nhóm >= rare_min_freq, còn lại gom "Other"; NaN -> "Missing"
        self.cat_levels_, self.onehot_levels_ = {}, {}
        for c in self.multi_cat_:
            freq = X[c].value_counts(normalize=True)
            kept = sorted(freq[freq >= self.rare_min_freq].index.tolist())
            self.cat_levels_[c] = kept
            levels = list(kept)
            if len(kept) < len(freq):          # có nhóm hiếm bị gom
                levels.append("Other")
            if X[c].isna().any():              # có giá trị thiếu
                levels.append("Missing")
            self.onehot_levels_[c] = levels

        # 7) Binary numeric (cờ 0/1): NaN -> mode
        self.binary_num_mode_ = {c: X[c].mode().iloc[0] if X[c].notna().any() else 0 for c in self.binary_num_}

        # 8) Numeric liên tục: ngưỡng cắt ngoại lệ, median, cột cần cờ missing
        lo_q, hi_q = self.winsor_quantiles
        self.clip_lo_ = X[self.cont_num_].quantile(lo_q)
        self.clip_hi_ = X[self.cont_num_].quantile(hi_q)
        self.median_ = X[self.cont_num_].clip(self.clip_lo_, self.clip_hi_, axis=1).median()
        self.flag_cols_ = [c for c in self.cont_num_ if miss_rate[c] >= self.missing_flag_threshold]

        # 9) Mean/std để scale (tính sau khi cắt ngoại lệ và impute)
        tmp = X[self.cont_num_].clip(self.clip_lo_, self.clip_hi_, axis=1).fillna(self.median_)
        self.mean_ = tmp.mean()
        self.std_ = tmp.std().replace(0, 1.0)

        # 10) Lưu tên cột đầu ra để bảo đảm thứ tự cố định
        self.feature_names_ = self.transform(X.head(5), _fitting=True).columns.tolist()
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame, scale: bool = False, _fitting: bool = False) -> pd.DataFrame:
        X = X.drop(columns=[c for c in self.drop_cols_ if c in X.columns]).copy()
        parts = []

        # numeric liên tục: cờ missing -> cắt ngoại lệ -> impute median
        cont = X[self.cont_num_]
        flags = cont[self.flag_cols_].isna().astype(np.int8)
        flags.columns = [f"{c}_ISNULL" for c in self.flag_cols_]
        cont = cont.clip(self.clip_lo_, self.clip_hi_, axis=1).fillna(self.median_)
        if scale:
            cont = (cont - self.mean_) / self.std_
        parts += [cont, flags]

        # cờ nhị phân dạng số
        bn = X[self.binary_num_].fillna(pd.Series(self.binary_num_mode_)).astype(np.int8)
        parts.append(bn)

        # binary categorical -> 0/1
        if self.binary_cat_:
            bc = pd.DataFrame({
                c: X[c].map(self.binary_map_[c]).fillna(self.binary_cat_mode_[c]).astype(np.int8)
                for c in self.binary_cat_
            }, index=X.index)
            parts.append(bc)

        # categorical nhiều mức -> one-hot (nhóm hiếm gom "Other", thiếu = "Missing")
        for c in self.multi_cat_:
            s = X[c].where(X[c].isin(self.cat_levels_[c]) | X[c].isna(), "Other").fillna("Missing")
            oh = pd.DataFrame({f"{c}__{lv}": (s == lv).astype(np.int8) for lv in self.onehot_levels_[c]},
                              index=X.index)
            parts.append(oh)

        out = pd.concat(parts, axis=1)
        if not _fitting:
            # đồng bộ cột với lúc fit (thêm cột one-hot vắng mặt = 0, bỏ cột lạ)
            out = out.reindex(columns=self.feature_names_, fill_value=0)
        return out

    def fit_transform(self, X, scale=False):
        return self.fit(X).transform(X, scale=scale)

    # ------------------------------------------------------- từ điển đặc trưng
    def feature_dictionary(self, raw_columns) -> pd.DataFrame:
        """Bảng mô tả từng cột đầu ra: loại xử lý + có phải đặc trưng tự tạo không."""
        rows = []
        raw = set(raw_columns)
        for f in self.feature_names_:
            if f.endswith("_ISNULL"):
                kind, src = "missing_flag", f[: -len("_ISNULL")]
            elif "__" in f and f.split("__")[0] in self.multi_cat_:
                kind, src = "one_hot", f.split("__")[0]
            elif f in self.cont_num_:
                kind, src = "numeric", f
            elif f in self.binary_cat_:
                kind, src = "binary_categorical", f
            else:
                kind, src = "binary_flag", f
            rows.append({"feature": f, "kind": kind, "source_column": src,
                         "engineered": src not in raw})
        return pd.DataFrame(rows)
