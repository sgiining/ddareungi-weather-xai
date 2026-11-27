"""[실험 1] 랜덤 포레스트 설정값 탐색 — 조정 단계(2023년 학습 → 2024년 평가)만 사용한다.

실행: 프로젝트 폴더에서  py experiments/01_tune.py
     (먼저 py build_dataset.py 로 data/processed/dataset.csv 를 만든다)
점검: py experiments/01_tune.py --quick   (작은 격자·시드 2개로 코드만 확인)

입력: 월·요일 원-핫 (common.make_X). 입력 열 수는 B1 23개, P 32개.

탐색 대상 (eda/DECISIONS.md 7·8번)
- n_estimators     : 200, 500
- max_depth        : None(제한 없음), 8, 16
- min_samples_leaf : 1, 3, 5
- max_features     : 1/3, 1/2, 2/3 (입력 열 수에 대한 비율). 1.0은 배깅이 되므로 넣지 않는다.
- 정답             : 원래 값, log(1+y)  (MAE는 항상 원래 단위로 계산)
→ 108개 조합 × 시드 5개, 달력 RF(B1)와 기상 추가 RF(P)에 각각 적용

선택 규칙
1) 시드 5개 MAE 평균이 가장 낮은 조합을 찾는다.
2) 그 조합과의 차이가 그 조합의 시드 표준편차 이내면 '성능이 같다'고 본다.
3) 성능이 같은 조합 중에서 아래 순서로 고른다.
   정답 원래 값 → n_estimators 작은 것 → max_depth 작은 것(None이 가장 큼)
   → min_samples_leaf 큰 것 → max_features 작은 것 → MAE 평균 낮은 것

결과
- results/tables/01_tune_B1.csv, 01_tune_P.csv : 조합별 MAE 평균·표준편차
- results/tables/01_tune_selected.json         : 모델별 선택 설정 (이후 실험이 읽어 씀)
"""

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common import MODEL_FEATURES, SEEDS, TABLES, ensure_dirs, get_split, load_dataset, make_X

GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 3, 5],
    "max_features": [1 / 3, 1 / 2, 2 / 3],
    "target": ["raw", "log1p"],
}
QUICK_GRID = {
    "n_estimators": [50],
    "max_depth": [None, 8],
    "min_samples_leaf": [1],
    "max_features": [1 / 3, 2 / 3],
    "target": ["raw", "log1p"],
}


def make_model(params, seed):
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )


def fit_predict(params, seed, X_tr, y_tr, X_ev):
    model = make_model(params, seed)
    if params["target"] == "log1p":
        model.fit(X_tr, np.log1p(y_tr))
        return np.expm1(model.predict(X_ev))
    model.fit(X_tr, y_tr)
    return model.predict(X_ev)


def search(X_tr, y_tr, X_ev, y_ev, grid, seeds):
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        start = time.time()
        maes = [mean_absolute_error(y_ev, fit_predict(params, s, X_tr, y_tr, X_ev)) for s in seeds]
        rows.append({**params,
                     "n_columns": X_tr.shape[1],
                     "n_features_per_split": max(1, int(params["max_features"] * X_tr.shape[1])),
                     "mae_mean": float(np.mean(maes)),
                     "mae_std": float(np.std(maes)),
                     "seconds": round(time.time() - start, 1)})
    return pd.DataFrame(rows)


def select(result: pd.DataFrame) -> dict:
    best = result.loc[result["mae_mean"].idxmin()]
    tied = result[result["mae_mean"] - best["mae_mean"] <= best["mae_std"]].copy()
    tied["_target"] = (tied["target"] != "raw").astype(int)
    tied["_depth"] = tied["max_depth"].fillna(np.inf)
    tied = tied.sort_values(
        ["_target", "n_estimators", "_depth", "min_samples_leaf", "max_features", "mae_mean"],
        ascending=[True, True, True, False, True, True])
    chosen = tied.iloc[0].drop(["_target", "_depth"]).to_dict()

    out = {}
    for k, v in chosen.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    for k in ("n_estimators", "min_samples_leaf", "n_columns", "n_features_per_split"):
        out[k] = int(out[k])
    if out["max_depth"] is not None:
        out["max_depth"] = int(out["max_depth"])
    out["n_tied"] = int(len(tied))
    out["best_mae_mean"] = float(best["mae_mean"])
    out["best_mae_std"] = float(best["mae_std"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset.csv 경로 (기본: data/processed/dataset.csv)")
    ap.add_argument("--quick", action="store_true", help="작은 격자·시드 2개로 코드만 점검")
    args = ap.parse_args()

    grid = QUICK_GRID if args.quick else GRID
    seeds = SEEDS[:2] if args.quick else SEEDS
    ensure_dirs()
    train, valid = get_split(load_dataset(args.data), "tune")
    n_combo = int(np.prod([len(v) for v in grid.values()]))
    print(f"조정 단계: 학습 {len(train)}일 → 평가 {len(valid)}일, 조합 {n_combo}개 × 시드 {len(seeds)}개")

    selected = {}
    for name, features in MODEL_FEATURES.items():
        X_tr, X_ev = make_X(train, features), make_X(valid, features)
        result = search(X_tr, train["rentals"], X_ev, valid["rentals"], grid, seeds)
        result = result.sort_values("mae_mean")
        suffix = "_quick" if args.quick else ""
        result.to_csv(TABLES / f"01_tune_{name}{suffix}.csv", index=False, encoding="utf-8-sig")
        selected[name] = select(result)
        print(f"\n[{name}] 입력 열 {X_tr.shape[1]}개")
        print(result.head(10).to_string(index=False))
        print(f"→ 선택: {selected[name]}")

    out_name = "01_tune_selected_quick.json" if args.quick else "01_tune_selected.json"
    with open(TABLES / out_name, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {TABLES / out_name}")


if __name__ == "__main__":
    main()
