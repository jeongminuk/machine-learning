# ============================================
# [초기 원형] 공행성(pair) + 다음달(value) 예측
# - 순수 lag 상관 기반
# - classifier / pseudo-label 없음
# - 0.34 근처 점수 나오던 구조
# ============================================

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# ===============================
# 0) 설정
# ===============================
MAX_LAG = 6
PAIR_TOP_K = 3000
BLEND_ALPHA = 0.9
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===============================
# 1) 데이터 로딩 & pivot
# ===============================
def load_pivot(train_path="train.csv"):
    df = pd.read_csv(train_path)
    monthly = df.groupby(["item_id", "year", "month"], as_index=False)["value"].sum()
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    pivot = monthly.pivot(index="item_id", columns="ym", values="value")
    pivot = pivot.fillna(0).sort_index(axis=1)
    return pivot

# ===============================
# 2) 안전한 상관
# ===============================
def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# ===============================
# 3) lag 상관 계산
# ===============================
def best_lag_corr(a, b, max_lag):
    best_corr = 0.0
    best_lag = 1
    for lag in range(1, max_lag + 1):
        if len(a) <= lag:
            continue
        c = safe_corr(a[:-lag], b[lag:])
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag
    return best_corr, best_lag

# ===============================
# 4) pair 전수 탐색 (순수 상관)
# ===============================
def find_pairs(pivot):
    items = pivot.index.tolist()
    rows = []

    for i in tqdm(items, desc="find_pairs"):
        a = pivot.loc[i].values
        for j in items:
            if i == j:
                continue
            b = pivot.loc[j].values
            corr, lag = best_lag_corr(a, b, MAX_LAG)
            rows.append({
                "leading_item_id": i,
                "following_item_id": j,
                "corr": corr,
                "lag": lag
            })

    df = pd.DataFrame(rows)
    df["abs_corr"] = df["corr"].abs()
    return df.sort_values("abs_corr", ascending=False)

# ===============================
# 5) 회귀 데이터 생성
# ===============================
def build_reg_data(pivot, pairs):
    X, y, meta = [], [], []

    for r in pairs.itertuples(index=False):
        a = pivot.loc[r.leading_item_id].values
        b = pivot.loc[r.following_item_id].values
        lag = r.lag

        for t in range(lag + 1, len(b) - 1):
            X.append([
                b[t],
                b[t-1],
                a[t-lag]
            ])
            y.append(b[t+1])
            meta.append((r.leading_item_id, r.following_item_id))

    return np.array(X), np.array(y), meta

# ===============================
# 6) MA3 baseline
# ===============================
def ma3(b, t):
    if t < 2:
        return b[t]
    return (b[t] + b[t-1] + b[t-2]) / 3

# ===============================
# 7) 전체 실행
# ===============================
def run_submission(train_path="train.csv", out_path="submission.csv"):
    pivot = load_pivot(train_path)
    pairs = find_pairs(pivot)

    # 🔥 상관 상위 K개만 사용
    pairs_sel = pairs.head(PAIR_TOP_K)

    X, y, _ = build_reg_data(pivot, pairs_sel)

    reg = LinearRegression()
    reg.fit(X, y)

    last_idx = pivot.shape[1] - 1
    rows = []

    for r in pairs_sel.itertuples(index=False):
        a = pivot.loc[r.leading_item_id].values
        b = pivot.loc[r.following_item_id].values
        lag = r.lag
        t = last_idx

        if t - lag < 0:
            continue

        x = np.array([[b[t], b[t-1], a[t-lag]]])
        pred = reg.predict(x)[0]
        base = ma3(b, t)

        pred = BLEND_ALPHA * pred + (1 - BLEND_ALPHA) * base
        pred = max(pred, 0)

        rows.append({
            "leading_item_id": r.leading_item_id,
            "following_item_id": r.following_item_id,
            "value": int(round(pred))
        })

    pd.DataFrame(rows).drop_duplicates(
        ["leading_item_id", "following_item_id"]
    ).to_csv(out_path, index=False)

    print("Saved:", out_path)

# ===============================
if __name__ == "__main__":
    run_submission("data/train.csv", "output/baseline_corr_034.csv")
