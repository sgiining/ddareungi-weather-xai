"""오차가 큰 날짜 선정(0-6)과 최고·최저기온 동시 변화 곡선(0-5) 계산.

모델 학습 코드에서 가져다 쓴다. 날짜·정답·예측값·입력 표만 있으면 동작한다.
"""

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826   # 정규분포에서 MAD를 표준편차와 같은 척도로 맞추는 상수
Z_THRESHOLD = 3.0    # 강건 z 점수 기준
N_SHOW_PER_SIDE = 3  # 개별 SHAP 그림: 과대예측·과소예측 각각 상위 몇 일


# ---------------------------------------------------------------------------
# 0-6. 오차가 큰 날짜 선정
# ---------------------------------------------------------------------------
def select_large_error_days(dates, y_true, pred_by_seed, extra=None):
    """테스트 기간 하나에서 오차가 큰 날짜를 고른다.

    dates        : 날짜 (길이 n)
    y_true       : 실제 대여건수 (길이 n)
    pred_by_seed : 시드별 예측값, shape (n, 시드 수). 원래 단위(건)로 넣는다.
    extra        : 결과 표에 함께 붙일 열 (예: is_offday, season, rain_mm)

    절차
    1) 시드 평균 예측으로 오차 e = y - ŷ 계산 (표 10과 같은 예측)
    2) 기간 전체의 치우침(중앙값)을 빼서 그날만의 오차 r = e - median(e) 계산
       → 2025년처럼 대여 수준 자체가 낮아진 기간에서 '모든 날의 과대예측'이
         선정 결과를 덮지 않게 한다. 치우침 크기는 ME로 따로 보고한다.
    3) 강건 z = r / (1.4826 × MAD(r)), |z| > 3 인 날을 '오차가 큰 날'로 선정
    4) 개별 SHAP 그림은 과대예측(z<0)·과소예측(z>0) 각각 |z| 상위 3일
    """
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred_by_seed, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    y_hat = pred.mean(axis=1)

    e = y - y_hat
    bias = np.median(e)
    r = e - bias
    mad = np.median(np.abs(r - np.median(r)))
    scale = MAD_SCALE * mad
    z = r / scale if scale > 0 else np.zeros_like(r)

    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "actual": y,
        "pred_mean": y_hat,
        "pred_seed_std": pred.std(axis=1),
        "error": e,
        "error_minus_bias": r,
        "robust_z": z,
    })
    if extra is not None:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(extra).reset_index(drop=True)], axis=1)

    out["direction"] = np.where(z < 0, "over", "under")   # over = 실제보다 많게 예측
    out["flagged"] = np.abs(z) > Z_THRESHOLD
    out["abs_z"] = np.abs(z)

    flagged = out[out["flagged"]].sort_values("abs_z", ascending=False)
    show = flagged.groupby("direction").head(N_SHOW_PER_SIDE)
    summary = {
        "n_days": len(out),
        "median_error(bias)": float(bias),
        "robust_scale": float(scale),
        "n_flagged": int(out["flagged"].sum()),
        "n_flagged_over": int((out["flagged"] & (out["direction"] == "over")).sum()),
        "n_flagged_under": int((out["flagged"] & (out["direction"] == "under")).sum()),
    }
    return flagged.drop(columns="abs_z"), show.drop(columns="abs_z"), summary


# ---------------------------------------------------------------------------
# 0-5. 최고·최저기온 동시 변화 곡선
# ---------------------------------------------------------------------------
def temperature_ice(predict_fn, X_eval, X_train, step=0.5,
                    tmax="max_temp", tmin="min_temp"):
    """최고·최저기온을 같은 폭으로 옮긴 예측 곡선.

    predict_fn : DataFrame -> 대여건수(건) 예측. 로그 정답 모델이면 expm1까지 해서 넘긴다.
    X_eval     : 테스트 기간 입력 (날짜당 1행)
    X_train    : 해당 테스트 기간의 학습 입력 (관측 범위 계산용)

    가로축은 '옮긴 뒤의 최고기온' x 이다. 날짜 i의 일교차 g_i = Tmax_i - Tmin_i 는 유지한다.
    (x 로 옮기는 것 = 두 기온에 같은 폭 δ = x - Tmax_i 를 더하는 것)
    날짜 i에서 x 가 허용되는 조건 (두 기온이 모두 학습 관측 범위 안):
        max(Lmax, Lmin + g_i) <= x <= min(Umax, Umin + g_i)

    평균 곡선은 centered ICE[10] 방식으로 계산한다.
    각 날짜의 곡선에서 기준점 x0(학습 기간 최고기온 중앙값, 모든 날짜에서 허용됨)의
    예측값을 뺀 뒤, 그 x 에서 허용되는 날짜끼리 평균한다.
    → 높은 기온 쪽에서 일교차가 작은 날(흐리거나 비 온 날)이 빠지면서
      평균이 달라 보이는 효과를 줄인다.
    """
    Lmax, Umax = X_train[tmax].min(), X_train[tmax].max()
    Lmin, Umin = X_train[tmin].min(), X_train[tmin].max()
    grid = np.round(np.arange(np.ceil(Lmax / step) * step, Umax + 1e-9, step), 4)
    x0 = float(np.round(X_train[tmax].median() / step) * step)
    if x0 not in grid:
        grid = np.sort(np.append(grid, x0))

    X_eval = X_eval.reset_index(drop=True)
    gap = (X_eval[tmax] - X_eval[tmin]).to_numpy()
    lo = np.maximum(Lmax, Lmin + gap)
    hi = np.minimum(Umax, Umin + gap)
    valid = (grid[None, :] >= lo[:, None] - 1e-9) & (grid[None, :] <= hi[:, None] + 1e-9)

    n, m = len(X_eval), len(grid)
    big = X_eval.loc[np.repeat(np.arange(n), m)].reset_index(drop=True)
    xs = np.tile(grid, n)
    big[tmax] = xs
    big[tmin] = xs - np.repeat(gap, m)
    pred = np.asarray(predict_fn(big), dtype=float).reshape(n, m)
    pred[~valid] = np.nan

    curves = pd.DataFrame(pred, columns=grid)
    j0 = int(np.where(grid == x0)[0][0])
    centered = curves.sub(curves.iloc[:, j0], axis=0)
    mean_curve = pd.DataFrame({
        "max_temp": grid,
        "centered_mean": centered.mean(axis=0, skipna=True).to_numpy(),
        "n_valid_days": valid.sum(axis=0),
        "share_valid": valid.mean(axis=0),
    })
    info = {"x0_reference": x0, "train_max_temp_range": (Lmax, Umax),
            "train_min_temp_range": (Lmin, Umin), "step": step}
    return curves, centered, mean_curve, info
