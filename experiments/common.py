"""모델링 실험 공통 설정과 함수 (experiments/ 의 모든 스크립트가 가져다 쓴다).

- 데이터 불러오기, 단계별 학습/평가 분할, 남은 빈칸 채우기
- 모델 입력 만들기: 월·요일 원-핫 인코딩 (모든 모델 같은 입력, eda/DECISIONS.md 8번)
- SHAP 값을 원래 변수(월·요일) 단위와 날씨 묶음 단위로 합치기
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # 프로젝트 폴더
sys.path.insert(0, str(ROOT))
from build_dataset import CALENDAR_FEATURES, WEATHER_FEATURES, WEATHER_GROUPS  # noqa: E402

DATASET = ROOT / "data" / "processed" / "dataset.csv"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

SEEDS = [0, 1, 2, 3, 4]
STAGES = ["tune", "test_a", "test_b"]   # dataset.csv 의 단계 열 이름

# 모델별 입력 (원래 변수 기준)
MODEL_FEATURES = {
    "B1": CALENDAR_FEATURES,                     # 달력 랜덤 포레스트
    "P": CALENDAR_FEATURES + WEATHER_FEATURES,   # 기상 추가 랜덤 포레스트 (제안 모델)
}

# 원-핫으로 바꾸는 변수와 범주 (범주를 하나도 빼지 않는다)
ONEHOT = {
    "month": list(range(1, 13)),
    "dow": list(range(7)),        # 0=월 ... 6=일
}


def ensure_dirs():
    for d in (TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def load_dataset(path=None) -> pd.DataFrame:
    path = Path(path) if path else DATASET
    if not path.exists():
        raise FileNotFoundError(f"{path} 이 없습니다. 먼저 py build_dataset.py 를 실행하세요.")
    df = pd.read_csv(path, parse_dates=["date"])
    for col in STAGES:
        df[col] = df[col].fillna("")
    return df


def get_split(df: pd.DataFrame, stage: str, last12: bool = False):
    """단계별 (학습, 평가) 표를 돌려준다.

    last12=True : 평가 연도 바로 전 해만 학습 (직전 12개월 보조 실험)
    날씨 빈칸(일조시간·풍속)은 학습 기간 중앙값으로만 채운다.
    """
    train = df[df[stage] == "train"].copy()
    evaluation = df[df[stage] == "eval"].copy()
    if last12:
        train = train[train["year"] == evaluation["year"].min() - 1].copy()
    for col in WEATHER_FEATURES:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        evaluation[col] = evaluation[col].fillna(med)
    return train, evaluation


def encoded_columns(raw_features) -> dict:
    """원래 변수 → 실제 입력 열 이름 목록. 예: month → month_1 ... month_12"""
    return {f: ([f"{f}_{v}" for v in ONEHOT[f]] if f in ONEHOT else [f]) for f in raw_features}


def make_X(df: pd.DataFrame, raw_features) -> pd.DataFrame:
    """모델 입력 표. 월·요일은 원-핫, 연도와 나머지는 그대로."""
    parts = []
    for f in raw_features:
        if f in ONEHOT:
            cats = ONEHOT[f]
            onehot = (df[f].to_numpy()[:, None] == np.array(cats)[None, :]).astype(int)
            parts.append(pd.DataFrame(onehot, columns=[f"{f}_{v}" for v in cats], index=df.index))
        else:
            parts.append(df[[f]])
    return pd.concat(parts, axis=1)


def shap_by_variable(shap_values, columns, raw_features) -> pd.DataFrame:
    """열 단위 SHAP을 원래 변수 단위로 합친다 (월 12열 → 월, 요일 7열 → 요일)."""
    sv = pd.DataFrame(np.asarray(shap_values), columns=list(columns))
    return pd.DataFrame({f: sv[cols].sum(axis=1) for f, cols in encoded_columns(raw_features).items()})


def shap_by_weather_group(shap_by_var: pd.DataFrame) -> pd.DataFrame:
    """변수 단위 SHAP을 날씨 묶음 단위로 부호 그대로 합친다 (계획서 3.4절)."""
    return pd.DataFrame({g: shap_by_var[cols].sum(axis=1) for g, cols in WEATHER_GROUPS.items()})
