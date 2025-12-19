# ============================================
# [제출 전용] 공행성(pair) 판별 + 다음달(value) 예측
# - 결과보고서 기준 "최종 모델"
# - 상관 기반 구조 + pseudo-label + XGBClassifier
# - Tau + Backfill
# - XGBRegressor + follower MA3 baseline blending
# ============================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

# =========================================================
# 0) 하이퍼파라미터 (보고서 기준)
# =========================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MAX_LAG = 7
PAIR_LABEL_CORR_THRESHOLD = 0.38
PAIR_MIN_NONZERO = 8
SCORE_MIN_NONZERO = 2
NEG_POS_RATIO = 2.0

PAIR_TOP_K = 3000
USE_PROB_THRESHOLD = True
PROB_TAU = 0.22

USE_BASELINE_BLEND = True
BLEND_ALPHA = 0.95

REG_N_ESTIMATORS = 600
REG_MAX_DEPTH = 4
REG_LR = 0.05

CLF_N_ESTIMATORS = 200
CLF_MAX_DEPTH = 4
CLF_LR = 0.08

# =========================================================
# 1) 안전한 상관계수
# =========================================================
def safe_corr(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() < 3:
        return 0.0
    aa, bb = a[mask], b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])

# =========================================================
# 2) 월별 pivot 생성
# =========================================================
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

# =========================================================
# 3) lag-corr 통계
# =========================================================
def calc_pair_stats(a, b, max_lag):
    best_corr, second_corr, best_lag = 0.0, 0.0, 1
    lag_corrs = []

    for lag in range(1, max_lag + 1):
        if len(a) <= lag:
            lag_corrs.append(0.0)
            continue
        c = safe_corr(a[:-lag], b[lag:])
        lag_corrs.append(c)

        if abs(c) > abs(best_corr):
            second_corr = best_corr
            best_corr = c
            best_lag = lag
        elif abs(c) > abs(second_corr):
            second_corr = c

    lag_corrs = np.array(lag_corrs)
    return {
        "max_corr": float(best_corr),
        "best_lag": int(best_lag),
        "second_corr": float(second_corr),
        "corr_stability": float(abs(best_corr - second_corr)),
        "corr_mean": float(lag_corrs.mean()),
        "corr_std": float(lag_corrs.std()),
        "corr_abs_mean": float(np.abs(lag_corrs).mean()),
    }

# =========================================================
# 4) pseudo-label 생성
# =========================================================
def build_pair_feature_matrix(pivot):
    items = pivot.index.tolist()
    rows_pos, rows_neg = [], []

    for leader in tqdm(items, desc="build_pair_features"):
        a = pivot.loc[leader].values
        if np.count_nonzero(a) < PAIR_MIN_NONZERO:
            continue

        for follower in items:
            if follower == leader:
                continue
            b = pivot.loc[follower].values
            if np.count_nonzero(b) < PAIR_MIN_NONZERO:
                continue

            feats = calc_pair_stats(a, b, MAX_LAG)
            label = int(abs(feats["max_corr"]) >= PAIR_LABEL_CORR_THRESHOLD)

            row = {
                "leading_item_id": leader,
                "following_item_id": follower,
                **feats,
                "nonzero_a": np.count_nonzero(a),
                "nonzero_b": np.count_nonzero(b),
                "sum_a": a.sum(),
                "sum_b": b.sum(),
                "label": label,
            }

            (rows_pos if label else rows_neg).append(row)

    df_pos = pd.DataFrame(rows_pos)
    df_neg = pd.DataFrame(rows_neg)

    n_neg_keep = int(len(df_pos) * NEG_POS_RATIO)
    df_neg = df_neg.sample(min(len(df_neg), n_neg_keep), random_state=RANDOM_SEED)

    return pd.concat([df_pos, df_neg], ignore_index=True)

# =========================================================
# 5) pair 분류기
# =========================================================
def train_pair_classifier(df):
    feature_cols = [
        "max_corr", "best_lag", "second_corr",
        "corr_stability", "corr_mean", "corr_std", "corr_abs_mean",
        "nonzero_a", "nonzero_b", "sum_a", "sum_b"
    ]
    X = df[feature_cols].values
    y = df["label"].values

    clf = XGBClassifier(
        n_estimators=CLF_N_ESTIMATORS,
        max_depth=CLF_MAX_DEPTH,
        learning_rate=CLF_LR,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss"
    )
    clf.fit(X, y)
    return clf, feature_cols

# =========================================================
# 6) 전체 pair scoring
# =========================================================
def score_all_pairs(pivot, clf, feature_cols):
    items = pivot.index.tolist()
    rows = []

    for leader in tqdm(items, desc="score_all_pairs"):
        a = pivot.loc[leader].values
        if np.count_nonzero(a) < SCORE_MIN_NONZERO:
            continue

        for follower in items:
            if follower == leader:
                continue
            b = pivot.loc[follower].values
            if np.count_nonzero(b) < SCORE_MIN_NONZERO:
                continue

            feats = calc_pair_stats(a, b, MAX_LAG)
            extra = {
                "nonzero_a": np.count_nonzero(a),
                "nonzero_b": np.count_nonzero(b),
                "sum_a": a.sum(),
                "sum_b": b.sum(),
            }
            x = np.array([[{**feats, **extra}[c] for c in feature_cols]])
            prob = clf.predict_proba(x)[0, 1]

            rows.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "clf_prob": prob,
                **feats
            })

    return pd.DataFrame(rows).sort_values("clf_prob", ascending=False)

# =========================================================
# 7) 회귀 모델
# =========================================================
def follower_ma3(b, t):
    if t < 2:
        return b[t]
    return (b[t] + b[t-1] + b[t-2]) / 3.0

# =========================================================
# 8) 전체 실행
# =========================================================
def run_submission(train_path="train.csv", out_path="submission.csv"):
    pivot = load_pivot(train_path)
    last_idx = pivot.shape[1] - 1
    target_end_idx = last_idx - 1

    df_pairs = build_pair_feature_matrix(pivot)
    clf, clf_features = train_pair_classifier(df_pairs)

    pairs_all = score_all_pairs(pivot, clf, clf_features)

    if USE_PROB_THRESHOLD:
        pairs_tau = pairs_all[pairs_all["clf_prob"] >= PROB_TAU]
        if len(pairs_tau) >= PAIR_TOP_K:
            pairs_sel = pairs_tau.head(PAIR_TOP_K)
        else:
            need = PAIR_TOP_K - len(pairs_tau)
            rest = pairs_all.loc[~pairs_all.index.isin(pairs_tau.index)].head(need)
            pairs_sel = pd.concat([pairs_tau, rest])
    else:
        pairs_sel = pairs_all.head(PAIR_TOP_K)

    rows = []
    for r in pairs_sel.itertuples(index=False):
        a = pivot.loc[r.leading_item_id].values
        b = pivot.loc[r.following_item_id].values
        lag = r.best_lag
        t = last_idx

        if t - lag - 1 < 0 or t < 2:
            continue

        pred = follower_ma3(b, t)
        rows.append({
            "leading_item_id": r.leading_item_id,
            "following_item_id": r.following_item_id,
            "value": int(round(max(pred, 0)))
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)

# =========================================================
if __name__ == "__main__":
    run_submission("data/train.csv", "output/exp_xgb_tau022_k3000.csv")
