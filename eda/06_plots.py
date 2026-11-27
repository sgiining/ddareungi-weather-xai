"""[점검 6] 계획서 4.1절에 쓸 수 있는 탐색 그림.

- 06_daily_rentals.png : 날짜별 대여건수 + 7일 이동평균 (전체 기간)
- 06_rentals_by_temp.png : 최고기온 구간별 하루 평균 대여건수 (평일/쉬는 날)
- 06_rentals_by_rain.png : 강수량 구간별 하루 평균 대여건수
- 06_rentals_by_dow.png : 요일별 하루 평균 대여건수

날씨와 대여건수의 관계 그림은 2023~2024년(조정 단계 자료)만 쓴다.

결과
- results/figures/06_*.png
- results/tables/06_*.csv
- results/sections/06_plots.md
"""

import pandas as pd

from eda_utils import DECISION_YEARS, FIGURES, TABLES, ensure_dirs, load_dataset, md_table, setup_matplotlib, write_section

TEMP_BINS = [-20, -5, 0, 5, 10, 15, 20, 25, 30, 35, 45]
RAIN_BINS = [-0.01, 0, 1, 5, 20, 50, 400]
RAIN_LABELS = ["0", "0~1", "1~5", "5~20", "20~50", "50 초과"]
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def main():
    ensure_dirs()
    plt = setup_matplotlib()
    df = load_dataset()
    d = df[df["year"].between(*DECISION_YEARS)].copy()

    # 1) 날짜별 대여건수
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["rentals"], lw=0.6, alpha=0.5, label="일별")
    ax.plot(df["date"], df["rentals"].rolling(7, center=True).mean(), lw=1.5, label="7일 이동평균")
    ax.set_ylabel("대여건수 (건)")
    ax.set_title("따릉이 일별 대여건수 (2023-01 ~ 2026-06)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_daily_rentals.png")
    plt.close(fig)

    # 2) 최고기온 구간별 (평일/쉬는 날)
    d["temp_bin"] = pd.cut(d["max_temp"], TEMP_BINS)
    by_temp = d.pivot_table(index="temp_bin", columns="is_offday", values="rentals",
                            aggfunc=["mean", "count"], observed=True)
    by_temp.columns = [f"{'평균' if a == 'mean' else '일수'}_{'쉬는날' if b else '평일'}" for a, b in by_temp.columns]
    by_temp.index = by_temp.index.astype(str)
    by_temp.to_csv(TABLES / "06_rentals_by_temp.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [("평균_평일", "평일"), ("평균_쉬는날", "쉬는 날")]:
        if col in by_temp:
            ax.plot(by_temp.index, by_temp[col], marker="o", label=label)
    ax.set_xlabel("최고기온 구간 (°C)")
    ax.set_ylabel("하루 평균 대여건수 (건)")
    ax.set_title(f"최고기온 구간별 평균 대여건수 ({DECISION_YEARS[0]}~{DECISION_YEARS[1]})")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_rentals_by_temp.png")
    plt.close(fig)

    # 3) 강수량 구간별
    d["rain_bin"] = pd.cut(d["rain_mm"], RAIN_BINS, labels=RAIN_LABELS)
    by_rain = d.groupby("rain_bin", observed=True)["rentals"].agg(["count", "mean"]).rename(
        columns={"count": "일수", "mean": "하루 평균"})
    by_rain.to_csv(TABLES / "06_rentals_by_rain.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_rain.index.astype(str), by_rain["하루 평균"])
    for i, (m, c) in enumerate(zip(by_rain["하루 평균"], by_rain["일수"])):
        ax.text(i, m, f"{c}일", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("일강수량 구간 (mm)")
    ax.set_ylabel("하루 평균 대여건수 (건)")
    ax.set_title(f"강수량 구간별 평균 대여건수 ({DECISION_YEARS[0]}~{DECISION_YEARS[1]})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_rentals_by_rain.png")
    plt.close(fig)

    # 4) 요일별
    by_dow = d.groupby("dow")["rentals"].mean().rename("하루 평균").to_frame()
    by_dow.index = [DOW_NAMES[i] for i in by_dow.index]
    by_dow.to_csv(TABLES / "06_rentals_by_dow.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(by_dow.index, by_dow["하루 평균"])
    ax.set_ylabel("하루 평균 대여건수 (건)")
    ax.set_title(f"요일별 평균 대여건수 ({DECISION_YEARS[0]}~{DECISION_YEARS[1]})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_rentals_by_dow.png")
    plt.close(fig)

    text = f"""
## 6. 탐색 그림

![일별 대여건수](figures/06_daily_rentals.png)

![최고기온 구간별](figures/06_rentals_by_temp.png)

{md_table(by_temp, 0)}

![강수량 구간별](figures/06_rentals_by_rain.png)

{md_table(by_rain, 0)}

![요일별](figures/06_rentals_by_dow.png)
"""
    write_section("06_plots", text)
    print("그림 4개 저장 완료:", FIGURES)


if __name__ == "__main__":
    main()
