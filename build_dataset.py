"""따릉이 일별 대여건수 + ASOS 서울 일자료 + 공휴일을 날짜로 합쳐 분석용 표를 만든다.

실행: py build_dataset.py

입력
- data/*일별 대여건수*.csv / .xlsx   : 따릉이 반기별 파일 (csv는 CP949, 2023년 상반기는 xlsx)
- data/raw/asos_seoul_daily_selected.csv : fetch_asos.py 결과
- data/raw/holidays_kr.csv               : fetch_holidays.py 결과

출력
- data/processed/dataset.csv : 하루 1행, 정답 + 달력 변수 + 날씨 변수 + 구간 표시
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

START = "2023-01-01"
END = "2026-06-30"

BIKE_PATTERN = os.path.join("data", "*일별 대여건수*")
ASOS_PATH = Path("data/raw/asos_seoul_daily_selected.csv")
HOLIDAY_PATH = Path("data/raw/holidays_kr.csv")
OUT_PATH = Path("data/processed/dataset.csv")

# 비·눈이 오지 않은 날 빈칸으로 오는 열 → 0으로 채운다 (데이터 점검에서 확인)
ZERO_FILL_COLS = ["rain_mm", "rain_hr", "new_snow_cm"]

# 모델에 넣을 변수 (2023→2024 조정 단계에서 결정. 모델링 코드에서 이 목록을 가져다 쓴다)
CALENDAR_FEATURES = ["year", "month", "dow", "is_weekend", "is_holiday", "is_offday"]
WEATHER_FEATURES = ["max_temp", "min_temp", "rain_mm", "rain_hr",
                    "wind_ms", "humidity", "sunshine_hr", "new_snow_cm"]
# 묶음 SHAP·ablation에 쓰는 날씨 묶음
WEATHER_GROUPS = {
    "temperature": ["max_temp", "min_temp"],
    "precipitation": ["rain_mm", "rain_hr"],
    "humidity_sun": ["humidity", "sunshine_hr"],
    "wind": ["wind_ms"],
    "snow": ["new_snow_cm"],
}


def read_bike_file(path: str) -> pd.DataFrame:
    """반기별 파일 하나를 읽어 [date, rentals] 두 열로 맞춘다."""
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")

    # 열 이름이 '대여일자' 또는 '대여일시'로 다르므로 위치로 맞춘다
    df = df.iloc[:, :2]
    df.columns = ["date", "rentals"]
    # 날짜가 '2023-01-01' 또는 '2023-01-01 00:00:00'으로 오므로 앞 10자리만 쓴다
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip().str[:10])
    df["rentals"] = pd.to_numeric(df["rentals"].astype(str).str.replace(",", ""), errors="coerce")
    print(f"  {os.path.basename(path)}: {len(df)}행 ({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df


def load_bike() -> pd.DataFrame:
    files = sorted(f for f in glob.glob(BIKE_PATTERN) if f.lower().endswith((".csv", ".xlsx")))
    if not files:
        raise FileNotFoundError(f"따릉이 파일을 찾지 못했습니다: {BIKE_PATTERN}")
    print(f"[따릉이] 파일 {len(files)}개")
    bike = pd.concat([read_bike_file(f) for f in files], ignore_index=True)

    dup = bike["date"].duplicated().sum()
    if dup:
        print(f"  주의: 중복 날짜 {dup}개 → 첫 값만 남김")
        bike = bike.drop_duplicates("date")
    return bike.sort_values("date").reset_index(drop=True)


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(ASOS_PATH, parse_dates=["date"])
    for col in ZERO_FILL_COLS:
        if col in w.columns:
            w[col] = w[col].fillna(0)
    return w


def load_holidays() -> pd.DataFrame:
    h = pd.read_csv(HOLIDAY_PATH, parse_dates=["date"])
    h = h[h["isHoliday"] == "Y"]
    return pd.DataFrame({"date": h["date"], "is_holiday": 1, "holiday_name": h["dateName"]})


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek  # 0=월 ... 6=일
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    # 쉬는 날 = 주말 또는 공휴일 (평일·휴일별 SHAP 비교에 사용)
    df["is_offday"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    # 계절 (집단별 SHAP 비교용, 모델 입력에는 넣지 않음)
    season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    df["season"] = df["month"].map(season_map)
    return df


def add_folds(df: pd.DataFrame) -> pd.DataFrame:
    """시간 순서를 지키며 학습 기간을 늘려 가는 평가 구간을 표시한다.

    tune   : 학습 2023        → 평가 2024        (설정값·변수 선택에만 사용)
    test_a : 학습 2023~2024   → 평가 2025        (성능 비교 + 주 설명 SHAP·ICE)
    test_b : 학습 2023~2025   → 평가 2026년 1~6월 (성능 비교 + 설명 안정성 확인)
    각 열의 값: "train" / "eval" / "" (해당 단계에서 쓰지 않음)
    """
    y = df["year"]
    df["tune"] = np.select([y == 2023, y == 2024], ["train", "eval"], default="")
    df["test_a"] = np.select([y <= 2024, y == 2025], ["train", "eval"], default="")
    df["test_b"] = np.select([y <= 2025, y == 2026], ["train", "eval"], default="")
    return df


def main():
    bike = load_bike()
    weather = load_weather()
    holidays = load_holidays()

    full = pd.DataFrame({"date": pd.date_range(START, END, freq="D")})
    df = full.merge(bike, on="date", how="left")
    df = df.merge(weather, on="date", how="left")
    df = df.merge(holidays, on="date", how="left")
    df = add_calendar(df)
    df = add_folds(df)

    calendar_cols = ["year", "month", "dow", "is_weekend", "is_holiday", "is_offday"]
    # avg_temp, cloud는 참고용으로 남겨 두지만 모델 입력에서는 제외한다 (WEATHER_FEATURES 참고)
    weather_cols = [c for c in weather.columns if c != "date"]
    info_cols = ["holiday_name", "season", "tune", "test_a", "test_b"]
    df = df[["date", "rentals"] + calendar_cols + weather_cols + info_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # ---- 점검 출력 ----
    print(f"\n[저장] {OUT_PATH}  ({len(df)}행 x {df.shape[1]}열)")
    print(f"기간 {df['date'].min().date()} ~ {df['date'].max().date()}, 기대 일수 {len(full)}")
    print("\n단계별 학습/평가 일수:")
    for col in ["tune", "test_a", "test_b"]:
        counts = df[col].value_counts()
        print(f"  {col:7s} train {counts.get('train', 0):4d}일 / eval {counts.get('eval', 0):4d}일")

    na = df[["rentals"] + weather_cols].isna().sum()
    na = na[na > 0]
    print("\n남은 결측 (모델링 때 학습 구간 값으로 채움):")
    print(na.to_string() if len(na) else "  없음")
    if df["rentals"].isna().any():
        print("  주의: 대여건수가 빈 날짜:", list(df.loc[df["rentals"].isna(), "date"].dt.date))

    print(f"\n공휴일 {int(df['is_holiday'].sum())}일, 쉬는 날(주말+공휴일) {int(df['is_offday'].sum())}일")

    print("\n연도별 하루 평균 대여건수:")
    print(df.groupby("year")["rentals"].mean().round(0).to_string())

    print("\n대여건수가 가장 적은 5일과 그날 날씨:")
    print(df.nsmallest(5, "rentals")[["date", "rentals", "rain_mm", "rain_hr", "new_snow_cm", "avg_temp", "holiday_name"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
