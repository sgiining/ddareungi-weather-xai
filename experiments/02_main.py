"""[실험 2] 본 실험 — 4개 모델 × 테스트 2기간의 예측 성능 (계획서 표 10)

실행: 프로젝트 폴더에서  py experiments/02_main.py
점검: py experiments/02_main.py --quick   (시드 2개, 트리 50개로 코드만 확인. 결과 파일에 _quick)

비교 모델 (계획서 표 3)
- 월·요일 평균  : 학습 기간에서 같은 월·요일의 평균 대여건수
- 선형회귀      : 달력 + 날씨 (원-핫 입력 32열, 기상 추가 RF와 같은 입력 표), 정답 원래 값
- 달력 RF (B1)  : 달력만 (23열), 01_tune_selected.json 의 B1 설정, 시드 5개
- 기상 추가 RF (P): 달력 + 날씨 (32열), 01_tune_selected.json 의 P 설정, 시드 5개

테스트 기간 (계획서 표 5)
- 2025   : 2023~2024년 학습 → 2025년 평가
- 2026H1 : 2023~2025년 학습 → 2026년 1~6월 평가

지표: MAE, RMSE, R², ME (원래 단위). 랜덤 포레스트는 시드별로 계산한 뒤 평균 ± 표준편차.
테스트 결과를 보고 입력·설정을 바꾸지 않는다 (계획서 3.3절). 한 번 돌린 결과를 그대로 보고한다.

결과 (results/tables/)
- 02_main_by_seed.csv     : 모델 × 기간 × 시드별 지표
- 02_main_summary.csv     : 모델 × 기간별 평균·표준편차 (표 10의 근거)
- 02_main_table10.csv     : 표 10에 옮길 형식 ("평균 ± 표준편차")
- 02_main_monthly_me.csv  : 모델 × 기간 × 월별 ME (대여 수준 변화에 따른 치우침 확인)
- 02_predictions.csv      : 날짜별 실제값·예측값 (5단계 오차 큰 날·SHAP 분석에서 사용)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common import (MODEL_FEATURES, PERIOD_LABEL, SEEDS, TABLES, TEST_STAGES, ensure_dirs,
                    fit_predict_rf, get_split, load_dataset, load_selected, make_X, metrics)

MODEL_NAMES = {   # 표 10의 순서와 이름
    "month_dow_mean": "월·요일 평균",
    "rf_calendar": "달력 랜덤 포레스트",
    "linear": "선형회귀",
    "rf_weather": "기상 추가 랜덤 포레스트",
}
PERIOD_NAMES = {"2025": "2025년", "2026H1": "2026년 상반기"}
METRICS = ["MAE", "RMSE", "R2", "ME"]


def predict_month_dow_mean(train: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    """같은 월·요일 평균. 학습에 없는 조합이 있으면 같은 월 평균 → 전체 평균 순으로 채운다."""
    by_md = train.groupby(["month", "dow"])["rentals"].mean()
    by_m = train.groupby("month")["rentals"].mean()
    pred = [by_md.get((m, d), by_m.get(m, train["rentals"].mean())) for m, d in zip(ev["month"], ev["dow"])]
    return np.asarray(pred, dtype=float)


def predict_linear(train, ev) -> np.ndarray:
    features = MODEL_FEATURES["P"]
    model = LinearRegression().fit(make_X(train, features), train["rentals"])
    return model.predict(make_X(ev, features))


def run_period(df, stage, selected, seeds, n_estimators):
    period = PERIOD_LABEL[stage]
    train, ev = get_split(df, stage)
    y = ev["rentals"].to_numpy()

    preds = ev[["date", "rentals", "month", "is_offday", "season"]].copy().reset_index(drop=True)
    preds.insert(1, "period", period)
    rows = []

    p = predict_month_dow_mean(train, ev)
    preds["pred_month_dow_mean"] = p
    rows.append({"model": "month_dow_mean", "period": period, "seed": np.nan, **metrics(y, p)})

    p = predict_linear(train, ev)
    preds["pred_linear"] = p
    rows.append({"model": "linear", "period": period, "seed": np.nan, **metrics(y, p)})

    for model_name, key in (("rf_calendar", "B1"), ("rf_weather", "P")):
        features = MODEL_FEATURES[key]
        X_tr, X_ev = make_X(train, features), make_X(ev, features)
        seed_preds = []
        for seed in seeds:
            _, p = fit_predict_rf(selected[key], seed, X_tr, train["rentals"], X_ev, n_estimators)
            preds[f"pred_{model_name}_s{seed}"] = p
            seed_preds.append(p)
            rows.append({"model": model_name, "period": period, "seed": seed, **metrics(y, p)})
        preds[f"pred_{model_name}_mean"] = np.mean(seed_preds, axis=0)   # 5단계 오차 분석용

    print(f"[{period}] 학습 {len(train)}일 → 평가 {len(ev)}일 완료")
    return pd.DataFrame(rows), preds


def summarize(by_seed: pd.DataFrame) -> pd.DataFrame:
    g = by_seed.groupby(["model", "period"])
    mean = g[METRICS].mean().add_suffix("_mean")
    std = g[METRICS].std(ddof=0).add_suffix("_std")      # 01_tune.py 와 같은 방식(np.std)
    out = pd.concat([mean, std], axis=1).reset_index()
    out["n_seeds"] = g["seed"].count().to_numpy()
    # 달력 RF 대비 MAE 변화율(%) — 음수면 오차가 줄어든 것
    base = out[out["model"] == "rf_calendar"].set_index("period")["MAE_mean"]
    out["MAE_vs_rf_calendar_pct"] = (out["MAE_mean"] / out["period"].map(base) - 1) * 100
    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    return out.sort_values(["period", "model"], key=lambda s: s.map(order) if s.name == "model" else s)


def table10(summary: pd.DataFrame) -> pd.DataFrame:
    def fmt(row, m):
        mean, std = row[f"{m}_mean"], row[f"{m}_std"]
        digits = 3 if m == "R2" else 1
        if row["n_seeds"] == 0:   # 시드가 없는 모델(월·요일 평균, 선형회귀)
            return f"{mean:,.{digits}f}"
        return f"{mean:,.{digits}f} ± {std:,.{digits}f}"

    rows = []
    for model in MODEL_NAMES:
        for period in PERIOD_NAMES:
            r = summary[(summary["model"] == model) & (summary["period"] == period)].iloc[0]
            rows.append({"모델": MODEL_NAMES[model], "테스트 기간": PERIOD_NAMES[period],
                         **{("R²" if m == "R2" else m): fmt(r, m) for m in METRICS}})
    return pd.DataFrame(rows)


def monthly_me(preds: pd.DataFrame) -> pd.DataFrame:
    cols = {"month_dow_mean": "pred_month_dow_mean", "linear": "pred_linear",
            "rf_calendar": "pred_rf_calendar_mean", "rf_weather": "pred_rf_weather_mean"}
    rows = []
    for (period, month), g in preds.groupby(["period", "month"]):
        for model, col in cols.items():
            rows.append({"period": period, "month": month, "model": model, "n_days": len(g),
                         "ME": float((g["rentals"] - g[col]).mean())})   # 시드 평균 예측의 ME = 시드별 ME의 평균
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset.csv 경로 (기본: data/processed/dataset.csv)")
    ap.add_argument("--quick", action="store_true", help="시드 2개, 트리 50개로 코드만 점검")
    args = ap.parse_args()

    seeds = SEEDS[:2] if args.quick else SEEDS
    n_estimators = 50 if args.quick else None
    suffix = "_quick" if args.quick else ""

    ensure_dirs()
    df = load_dataset(args.data)
    selected = load_selected()
    for key in ("B1", "P"):
        s = selected[key]
        print(f"{key} 설정: n_estimators={s['n_estimators']}, max_depth={s['max_depth']}, "
              f"min_samples_leaf={s['min_samples_leaf']}, max_features={s['max_features']:.3f}, target={s['target']}")

    results, all_preds = [], []
    for stage in TEST_STAGES:
        r, p = run_period(df, stage, selected, seeds, n_estimators)
        results.append(r)
        all_preds.append(p)
    by_seed = pd.concat(results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    summary = summarize(by_seed)
    t10 = table10(summary)
    by_seed.to_csv(TABLES / f"02_main_by_seed{suffix}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"02_main_summary{suffix}.csv", index=False, encoding="utf-8-sig")
    t10.to_csv(TABLES / f"02_main_table10{suffix}.csv", index=False, encoding="utf-8-sig")
    monthly_me(preds).to_csv(TABLES / f"02_main_monthly_me{suffix}.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(TABLES / f"02_predictions{suffix}.csv", index=False, encoding="utf-8-sig")

    print("\n표 10 (모델별 예측 성능)")
    print(t10.to_string(index=False))
    print("\n달력 랜덤 포레스트 대비 MAE 변화율(%)")
    print(summary[["model", "period", "MAE_vs_rf_calendar_pct"]].round(1).to_string(index=False))
    print(f"\n[저장] {TABLES}  (02_main_*{suffix}.csv, 02_predictions{suffix}.csv)")


if __name__ == "__main__":
    main()
