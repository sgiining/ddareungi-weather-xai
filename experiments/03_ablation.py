"""[실험 3] 기상 묶음 제거 실험 + 직전 12개월 학습 보조 실험 (계획서 표 4·표 11, 3.3절)

실행: 프로젝트 폴더에서  py experiments/03_ablation.py
점검: py experiments/03_ablation.py --quick   (시드 2개, 트리 50개로 코드만 확인. 결과 파일에 _quick)

A. 기상 묶음 제거 (표 11)
   - 기상 추가 RF(P)에서 날씨 묶음 5개를 하나씩 뺀 구성 + 전체 기상 변수 구성, 총 6개
   - 모두 P의 설정(01_tune_selected.json)을 그대로 쓴다. max_features는 같은 비율이다.
   - 학습·평가 기간은 표 10과 같다 (2025, 2026H1)
   - 같은 시드끼리 짝지어 "제거 구성 MAE − 전체 구성 MAE"를 계산한다 → 양수면 그 묶음이 도움이 된 것

B. 직전 12개월 학습 (보조 실험)
   - P를 각 테스트 기간 바로 전 해만으로 학습 (2025 ← 2024년, 2026H1 ← 2025년)
   - 전체 기간 학습과 MAE·RMSE·R²·ME를 비교한다
   - 학습 연도가 하나뿐이라 과거 대여 수준이 섞이지 않는다 → ME 변화로 과대예측의 원인을 확인

결과 (results/tables/)
- 03_ablation_by_seed.csv : 구성 × 기간 × 시드별 지표와 짝지은 MAE 차이
- 03_ablation_table11.csv : 표 11에 옮길 형식
- 03_last12_by_seed.csv   : 학습 기간 × 기간 × 시드별 지표
- 03_last12_table.csv     : 12개월 비교 표에 옮길 형식
"""

import argparse

import numpy as np
import pandas as pd

from common import (CALENDAR_FEATURES, PERIOD_LABEL, SEEDS, TABLES, TEST_STAGES, WEATHER_FEATURES,
                    WEATHER_GROUPS, ensure_dirs, fit_predict_rf, get_split, load_dataset, load_selected,
                    make_X, metrics)

GROUP_NAMES = {"temperature": "기온", "precipitation": "강수", "humidity_sky": "습도·하늘",
               "wind": "바람", "snow": "눈"}
PERIOD_NAMES = {"2025": "2025년", "2026H1": "2026년 상반기"}
METRICS = ["MAE", "RMSE", "R2", "ME"]


def ablation_configs():
    """(구성 id, 표에 쓸 이름, 날씨 변수 목록) — 계획서 표 4 순서"""
    configs = [("full", "전체 기상 변수", WEATHER_FEATURES)]
    for g, cols in WEATHER_GROUPS.items():
        configs.append((f"no_{g}", f"{GROUP_NAMES[g]} 제외", [c for c in WEATHER_FEATURES if c not in cols]))
    return configs


def run_ablation(df, params, seeds, n_estimators):
    rows = []
    for stage in TEST_STAGES:
        period = PERIOD_LABEL[stage]
        train, ev = get_split(df, stage)
        for cid, name, weather in ablation_configs():
            features = CALENDAR_FEATURES + weather
            X_tr, X_ev = make_X(train, features), make_X(ev, features)
            for seed in seeds:
                _, p = fit_predict_rf(params, seed, X_tr, train["rentals"], X_ev, n_estimators)
                rows.append({"config": cid, "name": name, "period": period, "seed": seed,
                             "n_weather": len(weather), "n_columns": X_tr.shape[1],
                             **metrics(ev["rentals"], p)})
        print(f"[묶음 제거 {period}] 6개 구성 완료")
    by_seed = pd.DataFrame(rows)
    full = by_seed[by_seed["config"] == "full"].set_index(["period", "seed"])["MAE"]
    by_seed["MAE_diff_vs_full"] = by_seed["MAE"] - [full[(p, s)] for p, s in zip(by_seed["period"], by_seed["seed"])]
    return by_seed


def table11(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, name, weather in ablation_configs():
        row = {"구성": name, "사용 기상 변수 수": len(weather)}
        for period, pname in PERIOD_NAMES.items():
            g = by_seed[(by_seed["config"] == cid) & (by_seed["period"] == period)]
            full_mae = by_seed[(by_seed["config"] == "full") & (by_seed["period"] == period)]["MAE"].mean()
            mae, mae_sd = g["MAE"].mean(), g["MAE"].std(ddof=0)
            d, d_sd = g["MAE_diff_vs_full"].mean(), g["MAE_diff_vs_full"].std(ddof=0)
            row[f"{pname} MAE"] = f"{mae:,.1f} ± {mae_sd:,.1f}"
            row[f"{pname} 전체 대비 변화"] = "–" if cid == "full" else \
                f"{d:+,.1f} ± {d_sd:,.1f} ({(mae / full_mae - 1) * 100:+.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def run_last12(df, params, seeds, n_estimators):
    features = CALENDAR_FEATURES + WEATHER_FEATURES
    rows = []
    for stage in TEST_STAGES:
        period = PERIOD_LABEL[stage]
        for train_kind, last12 in (("all", False), ("last12", True)):
            train, ev = get_split(df, stage, last12=last12)
            X_tr, X_ev = make_X(train, features), make_X(ev, features)
            years = f"{train['year'].min()}~{train['year'].max()}" if train["year"].nunique() > 1 \
                else str(train["year"].min())
            for seed in seeds:
                _, p = fit_predict_rf(params, seed, X_tr, train["rentals"], X_ev, n_estimators)
                rows.append({"train_kind": train_kind, "train_years": years, "n_train_days": len(train),
                             "period": period, "seed": seed, **metrics(ev["rentals"], p)})
        print(f"[직전 12개월 {period}] 완료")
    return pd.DataFrame(rows)


def last12_table(by_seed: pd.DataFrame) -> pd.DataFrame:
    kind_names = {"all": "전체 기간 학습", "last12": "직전 12개월 학습"}
    rows = []
    for period, pname in PERIOD_NAMES.items():
        for kind, kname in kind_names.items():
            g = by_seed[(by_seed["period"] == period) & (by_seed["train_kind"] == kind)]
            row = {"테스트 기간": pname, "학습": kname, "학습 연도": g["train_years"].iloc[0],
                   "학습 일수": int(g["n_train_days"].iloc[0])}
            for m in METRICS:
                digits = 3 if m == "R2" else 1
                row["R²" if m == "R2" else m] = f"{g[m].mean():,.{digits}f} ± {g[m].std(ddof=0):,.{digits}f}"
            rows.append(row)
    return pd.DataFrame(rows)


def check_against_main(by_seed: pd.DataFrame, suffix: str):
    """전체 기상 변수 구성이 02_main.py 의 기상 추가 RF와 같은 값인지 확인한다."""
    path = TABLES / f"02_main_by_seed{suffix}.csv"
    if not path.exists():
        print("  (02_main 결과가 없어 일치 확인을 건너뜀)")
        return
    main = pd.read_csv(path)
    main = main[main["model"] == "rf_weather"].set_index(["period", "seed"])["MAE"]
    full = by_seed[by_seed["config"] == "full"].set_index(["period", "seed"])["MAE"]
    common_idx = full.index.intersection(main.index)
    gap = (full[common_idx] - main[common_idx]).abs().max() if len(common_idx) else np.nan
    print(f"  02_main 기상 추가 RF와의 MAE 최대 차이: {gap:.6f}  (0이면 같은 모델)")


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
    params = load_selected()["P"]
    print(f"P 설정: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
          f"min_samples_leaf={params['min_samples_leaf']}, max_features={params['max_features']:.3f}, "
          f"target={params['target']}")

    abl = run_ablation(df, params, seeds, n_estimators)
    check_against_main(abl, suffix)
    t11 = table11(abl)
    abl.to_csv(TABLES / f"03_ablation_by_seed{suffix}.csv", index=False, encoding="utf-8-sig")
    t11.to_csv(TABLES / f"03_ablation_table11{suffix}.csv", index=False, encoding="utf-8-sig")

    l12 = run_last12(df, params, seeds, n_estimators)
    lt = last12_table(l12)
    l12.to_csv(TABLES / f"03_last12_by_seed{suffix}.csv", index=False, encoding="utf-8-sig")
    lt.to_csv(TABLES / f"03_last12_table{suffix}.csv", index=False, encoding="utf-8-sig")

    print("\n표 11 (기상 묶음 제거 실험)  변화 = 같은 시드끼리 '제거 MAE − 전체 MAE'의 평균 ± 표준편차")
    print(t11.to_string(index=False))
    print("\n직전 12개월 학습 비교")
    print(lt.to_string(index=False))
    print(f"\n[저장] {TABLES}  (03_ablation_*{suffix}.csv, 03_last12_*{suffix}.csv)")


if __name__ == "__main__":
    main()
