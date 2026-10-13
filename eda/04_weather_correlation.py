"""[점검 4] 날씨 후보 변수 10개 사이의 상관계수를 확인한다.

서로 거의 같은 정보를 담은 변수가 함께 들어가면 SHAP 기여가 나뉘어 해석이 흐려진다.
시험 기간 자료를 보지 않도록 2023~2024년(조정 단계 자료)만 쓴다.

- 피어슨 상관: 직선 관계의 강도
- 스피어먼 상관: 순서(크고 작음) 관계의 강도. 강수처럼 0이 많은 변수에 덜 민감
- |상관| ≥ 0.8 인 변수 쌍 목록

결과
- results/tables/04_*.csv
- results/figures/04_corr_pearson.png
- results/sections/04_correlation.md
"""

import numpy as np
import pandas as pd

from eda_utils import (DECISION_YEARS, FIGURES, KOREAN_NAME, TABLES, WEATHER_CANDIDATES, ensure_dirs,
                       load_dataset, md_table, setup_matplotlib, write_section)

THRESHOLD = 0.8


def high_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                rows.append({"변수 1": KOREAN_NAME[cols[i]], "변수 2": KOREAN_NAME[cols[j]], "상관계수": r})
    out = pd.DataFrame(rows, columns=["변수 1", "변수 2", "상관계수"])
    if len(out):
        out = out.reindex(out["상관계수"].abs().sort_values(ascending=False).index)
    return out


def main():
    ensure_dirs()
    df = load_dataset()
    d = df[df["year"].between(*DECISION_YEARS)]
    w = d[WEATHER_CANDIDATES]

    pearson = w.corr(method="pearson")
    spearman = w.corr(method="spearman")
    pairs_p = high_pairs(pearson, THRESHOLD)
    pairs_s = high_pairs(spearman, THRESHOLD)

    # 참고: 대여건수와의 상관 (조정 단계 자료)
    with_target = d[["rentals"] + WEATHER_CANDIDATES].corr(method="spearman")["rentals"].drop("rentals")
    with_target = with_target.rename("대여건수와의 스피어먼 상관").to_frame()
    with_target.index = [KOREAN_NAME[c] for c in with_target.index]

    named = lambda m: m.rename(index=KOREAN_NAME, columns=KOREAN_NAME)
    named(pearson).to_csv(TABLES / "04_corr_pearson.csv", encoding="utf-8-sig")
    named(spearman).to_csv(TABLES / "04_corr_spearman.csv", encoding="utf-8-sig")
    pairs_p.to_csv(TABLES / "04_high_pairs_pearson.csv", index=False, encoding="utf-8-sig")
    pairs_s.to_csv(TABLES / "04_high_pairs_spearman.csv", index=False, encoding="utf-8-sig")
    with_target.to_csv(TABLES / "04_corr_with_rentals.csv", encoding="utf-8-sig")

    # 그림: 피어슨 상관 히트맵
    plt = setup_matplotlib()
    labels = [KOREAN_NAME[c] for c in WEATHER_CANDIDATES]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pearson.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = pearson.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title(f"날씨 후보 변수 상관계수 (피어슨, {DECISION_YEARS[0]}~{DECISION_YEARS[1]})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_corr_pearson.png")
    plt.close(fig)

    text = f"""
## 4. 날씨 후보 변수 사이의 상관

자료: {DECISION_YEARS[0]}~{DECISION_YEARS[1]}년 {len(d):,}일 (시험 기간 제외)

![상관계수 히트맵](figures/04_corr_pearson.png)

### 4-1. |상관계수| ≥ {THRESHOLD} 인 변수 쌍 (피어슨)

{md_table(pairs_p, 2, index=False) if len(pairs_p) else "해당 없음"}

### 4-2. |상관계수| ≥ {THRESHOLD} 인 변수 쌍 (스피어먼)

{md_table(pairs_s, 2, index=False) if len(pairs_s) else "해당 없음"}

### 4-3. 참고: 각 날씨 변수와 대여건수의 스피어먼 상관

{md_table(with_target, 2)}

- 전체 상관표: `tables/04_corr_pearson.csv`, `tables/04_corr_spearman.csv`
- 기준값 {THRESHOLD}는 관례적인 값이다. 다른 기준을 쓰려면 이 파일의 THRESHOLD를 바꾼다.

> 판단 메모: 함께 넣지 않을 변수 쌍과 그 이유를 `eda/DECISIONS.md`의 4번 항목에 적는다.
"""
    write_section("04_correlation", text)
    print(text)


if __name__ == "__main__":
    main()
