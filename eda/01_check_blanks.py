"""[점검 1] ASOS의 강수·적설 빈칸이 정말 '비·눈이 오지 않은 날'인지 확인한다.

build_dataset.py 는 강수량(sumRn)·강수 계속시간(sumRnDur)·일 최심신적설(ddMefs)의
빈칸을 0으로 채운다. 이 처리가 맞는지 ASOS 원본(asos_seoul_daily_raw.csv)으로 확인한다.

확인 방법
1) 강수량 빈칸과 강수 계속시간 빈칸이 같은 날인지 (교차표)
2) 강수량이 빈 날에 일기현상(iscs)에 비·눈 기록이 있는지
3) 강수량이 빈 날에 9-9강수(n99Rn), 1시간 최다강수량(hr1MaxRn) 값이 있는지
4) 강수량이 0.0으로 '기록'된 날은 며칠인지 (빈칸과 0이 다른 뜻일 수 있으므로)
5) 신적설이 빈 날에 일기현상에 눈 기록이 있는지
6) 그 밖의 빈칸(풍속·일조시간 등)이 있는 날짜

결과
- results/tables/01_*.csv
- results/sections/01_blanks.md
- 판단이 애매한 날의 목록: results/tables/01_rain_blank_but_precip_signs.csv,
  01_snow_blank_but_snow_signs.csv  → 직접 열어 보고 판단한다.
"""

import re

import pandas as pd

from eda_utils import TABLES, ensure_dirs, load_asos_raw, load_dataset, md_table, write_section

RAIN_WORDS = ("비", "소나기", "우박")          # '진눈깨비', '안개비'도 '비'를 포함한다
SNOW_WORDS = ("눈",)                          # '진눈깨비', '싸락눈', '가루눈' 등 포함


def phenomenon_tokens(text) -> list:
    """일기현상 문자열에서 {…} 안의 현상 이름만 뽑는다. (시정·강도 표시는 제외)"""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"\{([^}]*)\}", text)
    return [t for t in tokens if not t.startswith(("시정", "강도")) and not re.match(r"^[\d.]+km$", t)]


def has_any(tokens, words) -> bool:
    return any(any(w in t for w in words) for t in tokens)


def main():
    ensure_dirs()
    raw = load_asos_raw()
    raw["tokens"] = raw["iscs"].apply(phenomenon_tokens)
    raw["rain_sign"] = raw["tokens"].apply(lambda t: has_any(t, RAIN_WORDS))
    raw["snow_sign"] = raw["tokens"].apply(lambda t: has_any(t, SNOW_WORDS))

    n = len(raw)
    rn_blank = raw["sumRn"].isna()
    dur_blank = raw["sumRnDur"].isna()
    snow_blank = raw["ddMefs"].isna()

    # 0) 일기현상에 나오는 현상 이름과 횟수 (키워드가 적절한지 직접 확인용)
    token_counts = (pd.Series([t for ts in raw["tokens"] for t in set(ts)])
                    .value_counts().rename_axis("현상").rename("기록된 날 수").to_frame())
    token_counts.to_csv(TABLES / "01_phenomenon_counts.csv", encoding="utf-8-sig")

    # 1) 강수량 빈칸 × 강수 계속시간 빈칸
    cross = pd.crosstab(rn_blank.map({True: "강수량 빈칸", False: "강수량 값 있음"}),
                        dur_blank.map({True: "강수시간 빈칸", False: "강수시간 값 있음"}))
    cross.index.name = "강수량 \\ 강수시간"
    cross.columns.name = None
    cross.to_csv(TABLES / "01_rain_vs_duration_blank.csv", encoding="utf-8-sig")

    # 2)~4) 강수량 빈칸인 날의 다른 근거
    rb = raw[rn_blank]
    rain_summary = pd.DataFrame({
        "날 수": [
            int(rn_blank.sum()),
            int((rb["rain_sign"]).sum()),
            int((rb["n99Rn"].fillna(0) > 0).sum()),
            int((rb["hr1MaxRn"].fillna(0) > 0).sum()),
            int((rb["sumRnDur"].fillna(0) > 0).sum()),
            int((raw["sumRn"] == 0).sum()),
        ]},
        index=[
            "강수량(sumRn)이 빈칸인 날",
            "  └ 그중 일기현상에 비·눈 기록이 있는 날",
            "  └ 그중 9-9강수(n99Rn) > 0 인 날",
            "  └ 그중 1시간 최다강수량(hr1MaxRn) > 0 인 날",
            "  └ 그중 강수 계속시간(sumRnDur) > 0 인 날",
            "참고: 강수량이 0.0으로 기록된 날",
        ])
    rain_summary.to_csv(TABLES / "01_rain_blank_summary.csv", encoding="utf-8-sig")

    suspicious_rain = rb[rb["rain_sign"] | (rb["n99Rn"].fillna(0) > 0) | (rb["sumRnDur"].fillna(0) > 0)]
    suspicious_rain[["date", "sumRn", "sumRnDur", "n99Rn", "hr1MaxRn", "iscs"]].to_csv(
        TABLES / "01_rain_blank_but_precip_signs.csv", index=False, encoding="utf-8-sig")

    # 5) 신적설 빈칸인 날
    sb = raw[snow_blank]
    snow_summary = pd.DataFrame({
        "날 수": [
            int(snow_blank.sum()),
            int(sb["snow_sign"].sum()),
            int((sb["ddMes"].fillna(0) > 0).sum()),
            int((raw["ddMefs"] == 0).sum()),
        ]},
        index=[
            "신적설(ddMefs)이 빈칸인 날",
            "  └ 그중 일기현상에 눈 기록이 있는 날",
            "  └ 그중 최심적설(ddMes, 쌓여 있던 눈) > 0 인 날",
            "참고: 신적설이 0.0으로 기록된 날",
        ])
    snow_summary.to_csv(TABLES / "01_snow_blank_summary.csv", encoding="utf-8-sig")
    sb[sb["snow_sign"]][["date", "ddMefs", "ddMes", "sumRn", "iscs"]].to_csv(
        TABLES / "01_snow_blank_but_snow_signs.csv", index=False, encoding="utf-8-sig")

    # 6) 그 밖의 빈칸 (0으로 채우지 않는 변수)
    ds = load_dataset()
    other_cols = ["avg_temp", "max_temp", "min_temp", "humidity", "sunshine_hr", "cloud", "wind_ms"]
    other = ds[["date"] + other_cols]
    other_blank = other[other[other_cols].isna().any(axis=1)]
    other_blank.to_csv(TABLES / "01_other_blank_days.csv", index=False, encoding="utf-8-sig")
    other_count = ds[other_cols].isna().sum().rename("빈칸 수").to_frame()

    text = f"""
## 1. 강수·적설 빈칸 점검

ASOS 원본 {n:,}일 기준. build_dataset.py는 강수량·강수 계속시간·신적설의 빈칸을 0으로 채운다.
아래 결과로 그 처리가 맞는지 판단한다.

### 1-1. 강수량 빈칸과 강수 계속시간 빈칸이 같은 날인가

{md_table(cross, 0)}

### 1-2. 강수량이 빈칸인 날에 비가 왔다는 다른 기록이 있는가

{md_table(rain_summary, 0)}

- 일기현상 키워드: 비 관련 {RAIN_WORDS}, 눈 관련 {SNOW_WORDS} (현상 이름에 이 글자가 들어가면 해당으로 봄)
- 9-9강수는 오전 9시~다음날 오전 9시 합계라서, 하루(0~24시) 강수량과 기간이 다르다.
- 근거가 겹치는 날의 상세 목록: `tables/01_rain_blank_but_precip_signs.csv`

### 1-3. 신적설이 빈칸인 날에 눈이 왔다는 기록이 있는가

{md_table(snow_summary, 0)}

- 상세 목록: `tables/01_snow_blank_but_snow_signs.csv`

### 1-4. 0으로 채우지 않는 변수의 빈칸

{md_table(other_count, 0)}

- 빈칸이 있는 날짜: `tables/01_other_blank_days.csv` (모델링 때 학습 기간 중앙값으로 채울 예정)

### 1-5. 일기현상에 기록된 현상 이름 (상위 15개)

{md_table(token_counts.head(15), 0)}

> 판단 메모: 결과를 보고 `eda/DECISIONS.md`의 1번 항목에 직접 적는다.
"""
    write_section("01_blanks", text)
    print(text)


if __name__ == "__main__":
    main()
