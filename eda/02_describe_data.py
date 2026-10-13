"""[점검 2] 데이터 기본 통계 (계획서 4.1 Dataset 절에 넣을 값).

- 변수별 개수·빈칸 수·최솟값·최댓값·평균·중앙값
- 연도별 공휴일 수, 쉬는 날(주말+공휴일) 수
- 단계(tune/test_a/test_b)별 학습·평가 일수

결과
- results/tables/02_*.csv
- results/sections/02_describe.md
"""

import pandas as pd

from eda_utils import KOREAN_NAME, TABLES, WEATHER_CANDIDATES, ensure_dirs, load_dataset, md_table, write_section


def main():
    ensure_dirs()
    df = load_dataset()

    cols = ["rentals"] + WEATHER_CANDIDATES
    stats = pd.DataFrame({
        "개수": df[cols].count(),
        "빈칸": df[cols].isna().sum(),
        "최솟값": df[cols].min(),
        "최댓값": df[cols].max(),
        "평균": df[cols].mean(),
        "중앙값": df[cols].median(),
    })
    stats.index = [f"{KOREAN_NAME.get(c, c)} ({c})" for c in stats.index]
    stats.to_csv(TABLES / "02_variable_stats.csv", encoding="utf-8-sig")

    cal = df.groupby("year").agg(
        일수=("date", "size"),
        공휴일=("is_holiday", "sum"),
        주말=("is_weekend", "sum"),
        쉬는날=("is_offday", "sum"),
    )
    cal.loc["전체"] = cal.sum()
    cal.to_csv(TABLES / "02_calendar_counts.csv", encoding="utf-8-sig")

    stage_rows = []
    for col in ["tune", "test_a", "test_b"]:
        s = df[col].fillna("")
        tr, ev = df[s == "train"], df[s == "eval"]
        stage_rows.append({
            "단계": col,
            "학습 기간": f"{tr['date'].min().date()} ~ {tr['date'].max().date()}",
            "학습 일수": len(tr),
            "평가 기간": f"{ev['date'].min().date()} ~ {ev['date'].max().date()}",
            "평가 일수": len(ev),
        })
    stages = pd.DataFrame(stage_rows).set_index("단계")
    stages.to_csv(TABLES / "02_stages.csv", encoding="utf-8-sig")

    text = f"""
## 2. 데이터 기본 통계

기간: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df):,}일 (하루 1행)

### 2-1. 변수별 통계 (강수·적설은 빈칸을 0으로 채운 뒤의 값)

{md_table(stats, 1)}

### 2-2. 연도별 공휴일·쉬는 날 수

{md_table(cal, 0)}

### 2-3. 단계별 학습·평가 기간

{md_table(stages, 0)}
"""
    write_section("02_describe", text)
    print(text)


if __name__ == "__main__":
    main()
