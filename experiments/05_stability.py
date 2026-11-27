"""[실험 5] 설명 안정성 — 상위 항목의 Jaccard 지수 (계획서 식 (5), 4.5.4절)

실행: 프로젝트 폴더에서  py experiments/05_stability.py
      (먼저 py experiments/04_shap.py 로 04_importance*.csv 를 만든다. 몇 초면 끝난다)

규칙 (eda/DECISIONS.md 7번)
- 상위 항목: 평균 |SHAP| 기준 상위 변수 5개, 상위 날씨 묶음 3개
- 비교
  1) 기간 간  : 2025 vs 2026H1 — 시드 평균 SHAP 기준 1쌍 + 같은 시드끼리 5쌍
  2) 시드 간  : 기간마다 시드 5개의 모든 쌍 10개
- J(A, B) = |A ∩ B| / |A ∪ B|. 1에 가까울수록 상위 항목이 많이 겹친다. 순위·크기 차이는 반영하지 않는다.
- 해석을 돕기 위해 k번째와 k+1번째 항목의 중요도 차이(경계 차이)를 함께 저장한다.
  경계 차이가 작으면 시드나 기간에 따라 상위 집합이 쉽게 바뀔 수 있다.

결과 (results/tables/)
- 05_stability.csv       : 비교별 Jaccard (쌍 수, 평균, 최소, 최대)
- 05_stability_table.csv : 계획서 표에 옮길 형식
- 05_top_sets.csv        : 기간 × (시드 평균, 시드별) 상위 항목 목록과 경계 차이
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from common import TABLES

TOP_K = {"variable": 5, "group": 3}
LEVEL_NAMES = {"variable": "상위 변수 5개", "group": "상위 날씨 묶음 3개"}
PERIOD_NAMES = {"2025": "2025년", "2026H1": "2026년 상반기"}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


def top_info(table: pd.DataFrame, level: str):
    """상위 k개 집합, 순서 있는 이름 목록, 경계 차이(k번째 − k+1번째 평균 |SHAP|)"""
    k = TOP_K[level]
    d = table[table["level"] == level].sort_values("mean_abs", ascending=False)
    names = list(d["name_ko"])
    gap = float(d["mean_abs"].iloc[k - 1] - d["mean_abs"].iloc[k]) if len(d) > k else np.nan
    return set(names[:k]), names[:k], gap


def summarize(rows):
    vals = np.array([r for r in rows if not np.isnan(r)])
    return {"pairs": len(vals), "mean": vals.mean(), "min": vals.min(), "max": vals.max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="04_shap.py --quick 결과(_quick)로 실행")
    args = ap.parse_args()
    suffix = "_quick" if args.quick else ""

    imp = pd.read_csv(TABLES / f"04_importance{suffix}.csv", dtype={"period": str})
    by_seed = pd.read_csv(TABLES / f"04_importance_by_seed{suffix}.csv", dtype={"period": str})
    periods = [p for p in PERIOD_NAMES if p in set(imp["period"])]
    seeds = sorted(by_seed["seed"].unique())

    # 상위 항목 목록
    top_rows, sets = [], {}
    for period in periods:
        for level in TOP_K:
            s, names, gap = top_info(imp[imp["period"] == period], level)
            sets[(period, "mean", level)] = s
            top_rows.append({"period": period, "source": "시드 평균", "level": level,
                             "top": ", ".join(names), "boundary_gap": gap})
            for seed in seeds:
                s, names, gap = top_info(by_seed[(by_seed["period"] == period) & (by_seed["seed"] == seed)], level)
                sets[(period, seed, level)] = s
                top_rows.append({"period": period, "source": f"시드 {seed}", "level": level,
                                 "top": ", ".join(names), "boundary_gap": gap})
    top_sets = pd.DataFrame(top_rows)

    # Jaccard
    rows = []
    a, b = periods[0], periods[-1]
    for level in TOP_K:
        rows.append({"comparison": "기간 간 (시드 평균 SHAP)", "level": level,
                     **summarize([jaccard(sets[(a, "mean", level)], sets[(b, "mean", level)])])})
        rows.append({"comparison": "기간 간 (같은 시드끼리)", "level": level,
                     **summarize([jaccard(sets[(a, s, level)], sets[(b, s, level)]) for s in seeds])})
        for period in periods:
            rows.append({"comparison": f"시드 간 ({PERIOD_NAMES[period]})", "level": level,
                         **summarize([jaccard(sets[(period, s1, level)], sets[(period, s2, level)])
                                      for s1, s2 in combinations(seeds, 2)])})
    stab = pd.DataFrame(rows)

    # 계획서 표 형식: 평균 (최소~최대, 쌍 수)
    def fmt(r):
        if r["pairs"] == 1:
            return f"{r['mean']:.3f}"
        return f"{r['mean']:.3f} ({r['min']:.3f}~{r['max']:.3f}, {int(r['pairs'])}쌍)"

    table = (stab.assign(value=stab.apply(fmt, axis=1))
                 .pivot(index="comparison", columns="level", values="value")
                 .reindex(stab["comparison"].drop_duplicates())
                 .rename(columns=LEVEL_NAMES)[list(LEVEL_NAMES.values())]
                 .reset_index().rename(columns={"comparison": "비교"}))

    stab.to_csv(TABLES / f"05_stability{suffix}.csv", index=False, encoding="utf-8-sig")
    table.to_csv(TABLES / f"05_stability_table{suffix}.csv", index=False, encoding="utf-8-sig")
    top_sets.to_csv(TABLES / f"05_top_sets{suffix}.csv", index=False, encoding="utf-8-sig")

    print("상위 항목 (시드 평균 SHAP)  경계 차이 = k번째 − k+1번째 평균 |SHAP|")
    print(top_sets[top_sets["source"] == "시드 평균"][["period", "level", "top", "boundary_gap"]]
          .round(4).to_string(index=False))
    print("\n설명 안정성 (Jaccard 지수: 평균 (최소~최대, 쌍 수))")
    print(table.to_string(index=False))
    changed = top_sets[top_sets["source"] != "시드 평균"].groupby(["period", "level"])["top"].nunique()
    print("\n기간별 시드 5개에서 나온 서로 다른 상위 집합(순서 포함) 수")
    print(changed.to_string())
    print(f"\n[저장] {TABLES}  (05_stability{suffix}.csv, 05_stability_table{suffix}.csv, 05_top_sets{suffix}.csv)")


if __name__ == "__main__":
    main()
