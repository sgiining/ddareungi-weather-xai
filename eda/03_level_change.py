"""[점검 3] 연도별 대여 수준이 달라졌는지 확인한다.

계절 차이가 섞이지 않도록 '같은 달끼리' 비교하고,
날씨·요일 차이가 섞이지 않도록 '평일이면서 비가 오지 않은 날'만 따로 비교한다.
원인(요금, 대여소 수 등)은 이 데이터에 없으므로 여기서는 '변화가 있는지'만 확인한다.

결과
- results/tables/03_*.csv
- results/figures/03_monthly_mean_by_year.png
- results/sections/03_level_change.md
"""

import pandas as pd

from eda_utils import FIGURES, TABLES, ensure_dirs, load_dataset, md_table, setup_matplotlib, write_section


def main():
    ensure_dirs()
    df = load_dataset()
    df["half"] = df["year"].astype(str) + "-" + df["month"].le(6).map({True: "상반기", False: "하반기"})

    # 1) 연도별·반기별 하루 평균
    by_year = df.groupby("year")["rentals"].agg(["count", "mean", "median"]).rename(
        columns={"count": "일수", "mean": "하루 평균", "median": "하루 중앙값"})
    by_half = df.groupby("half")["rentals"].agg(["count", "mean"]).rename(
        columns={"count": "일수", "mean": "하루 평균"})

    # 2) 같은 달끼리: 월 × 연도 하루 평균, 전년 같은 달 대비 변화율(%)
    pivot = df.pivot_table(index="month", columns="year", values="rentals", aggfunc="mean")
    yoy = (pivot / pivot.shift(axis=1) - 1) * 100
    yoy = yoy.drop(columns=pivot.columns[0])
    yoy.columns = [f"{c} vs {c - 1} (%)" for c in yoy.columns]

    # 3) 조건을 맞춘 비교: 평일(쉬는 날 아님) + 강수량 0인 날
    calm = df[(df["is_offday"] == 0) & (df["rain_mm"] == 0)]
    calm_by_year = calm.groupby("year")["rentals"].agg(["count", "mean"]).rename(
        columns={"count": "해당 일수", "mean": "하루 평균"})
    calm_pivot = calm.pivot_table(index="month", columns="year", values="rentals", aggfunc="mean")

    # 4) 연도별 날씨 요약 (날씨가 해마다 크게 달랐는지)
    weather_by_year = df.groupby("year").agg(
        평균_최고기온=("max_temp", "mean"),
        강수1mm이상_일수=("rain_mm", lambda s: int((s >= 1).sum())),
        연강수량=("rain_mm", "sum"),
        신적설_있는날=("new_snow_cm", lambda s: int((s > 0).sum())),
        일수=("date", "size"),
    )

    for name, t in [("03_by_year", by_year), ("03_by_half", by_half), ("03_month_by_year", pivot),
                    ("03_same_month_change_pct", yoy), ("03_calm_days_by_year", calm_by_year),
                    ("03_calm_days_month_by_year", calm_pivot), ("03_weather_by_year", weather_by_year)]:
        t.to_csv(TABLES / f"{name}.csv", encoding="utf-8-sig")

    # 그림: 월별 하루 평균을 연도별 선으로
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for year in pivot.columns:
        ax.plot(pivot.index, pivot[year], marker="o", label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("월")
    ax.set_ylabel("하루 평균 대여건수 (건)")
    ax.set_title("월별 하루 평균 대여건수 (연도별)")
    ax.grid(alpha=0.3)
    ax.legend(title="연도")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_monthly_mean_by_year.png")
    plt.close(fig)

    text = f"""
## 3. 연도별 대여 수준 변화

원인(요금, 대여소·자전거 수 등)은 이 데이터에 없다. 여기서는 변화가 있는지만 확인한다.

### 3-1. 연도별 하루 평균 (2026년은 1~6월만 있으므로 계절 차이가 섞임)

{md_table(by_year, 0)}

### 3-2. 반기별 하루 평균

{md_table(by_half, 0)}

### 3-3. 같은 달끼리 비교: 전년 같은 달 대비 변화율(%)

{md_table(yoy, 1)}

![월별 하루 평균](figures/03_monthly_mean_by_year.png)

### 3-4. 조건을 맞춘 비교: 평일이면서 비가 오지 않은 날만

{md_table(calm_by_year, 0)}

월 × 연도 표: `tables/03_calm_days_month_by_year.csv`

### 3-5. 연도별 날씨 요약 (날씨 차이로 설명될 수 있는지 참고)

{md_table(weather_by_year, 1)}

> 판단 메모: 수준 변화가 있다고 보는지, 그렇다면 계획서에 어떻게 쓸지 `eda/DECISIONS.md`의 3번 항목에 적는다.
"""
    write_section("03_level_change", text)
    print(text)


if __name__ == "__main__":
    main()
