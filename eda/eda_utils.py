"""탐색(EDA) 스크립트 공통 설정과 함수.

어느 폴더에서 실행해도 프로젝트 폴더를 기준으로 경로를 잡는다.
결과는 eda/results/ 아래에 저장한다.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # 프로젝트 폴더
DATASET = ROOT / "data" / "processed" / "dataset.csv"
ASOS_RAW = ROOT / "data" / "raw" / "asos_seoul_daily_raw.csv"
HOLIDAYS = ROOT / "data" / "raw" / "holidays_kr.csv"

RESULTS = Path(__file__).resolve().parent / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SECTIONS = RESULTS / "sections"

CALENDAR = ["year", "month", "dow", "is_weekend", "is_holiday", "is_offday"]

# 날씨 후보 10개 (dataset.csv 열 이름)
WEATHER_CANDIDATES = [
    "avg_temp", "max_temp", "min_temp",
    "rain_mm", "rain_hr",
    "humidity", "sunshine_hr", "cloud",
    "wind_ms",
    "new_snow_cm",
]

KOREAN_NAME = {
    "rentals": "대여건수",
    "avg_temp": "평균기온", "max_temp": "최고기온", "min_temp": "최저기온",
    "rain_mm": "일강수량", "rain_hr": "강수 계속시간",
    "humidity": "평균 상대습도", "sunshine_hr": "합계 일조시간", "cloud": "평균 전운량",
    "wind_ms": "평균 풍속", "new_snow_cm": "일 최심신적설",
}

# 변수 선택 판단에는 시험 기간(2025년 이후)을 쓰지 않는다
DECISION_YEARS = (2023, 2024)


def ensure_dirs():
    for d in (TABLES, FIGURES, SECTIONS):
        d.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET} 이 없습니다. 먼저 py build_dataset.py 를 실행하세요.")
    df = pd.read_csv(DATASET, parse_dates=["date"])
    return df


def load_asos_raw() -> pd.DataFrame:
    if not ASOS_RAW.exists():
        raise FileNotFoundError(f"{ASOS_RAW} 이 없습니다. 먼저 py api_code/fetch_asos.py 를 실행하세요.")
    raw = pd.read_csv(ASOS_RAW)
    raw["date"] = pd.to_datetime(raw["tm"])
    return raw


def md_table(df: pd.DataFrame, digits: int = 2, index: bool = True) -> str:
    """tabulate 없이 DataFrame을 마크다운 표로 바꾼다."""
    d = df.copy()
    d.columns.name = None
    if index:
        d.index.name = d.index.name or ""
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    # 연도·월처럼 '이름'에 해당하는 숫자 열은 쉼표 없이 쓴다
    plain = {c for c in cols if c in ("", "year", "month", "dow", "연도", "월")}

    def fmt(v, col):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            return f"{v:.{digits}f}" if col in plain else f"{v:,.{digits}f}"
        if isinstance(v, int) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("int")):
            return str(int(v)) if col in plain else f"{int(v):,}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v, c) for v, c in zip(row, cols)) + " |")
    return "\n".join(lines)


def write_section(name: str, text: str):
    """결과 보고서의 한 절을 results/sections/<name>.md 로 저장한다."""
    ensure_dirs()
    (SECTIONS / f"{name}.md").write_text(text.strip() + "\n", encoding="utf-8")


def setup_matplotlib():
    """한글이 깨지지 않도록 글꼴을 맞춘다 (Windows: 맑은 고딕)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return plt
