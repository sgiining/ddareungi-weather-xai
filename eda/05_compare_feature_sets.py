"""[점검 5] 날씨 변수 조합별 예측 오차를 비교해 최종 변수 선택의 근거를 만든다.

조정 단계(2023년 학습 → 2024년 평가)만 사용한다. 시험 A·B 기간은 보지 않는다.
모델: 랜덤 포레스트(기본 설정, 트리 300개). 설정값 튜닝 전이므로 '조합 사이의 상대 비교'로만 본다.
난수 시드 5개로 반복해 평균과 표준편차를 낸다.

조합은 아래 FEATURE_SETS에서 추가·수정할 수 있다.

결과
- results/tables/05_feature_set_mae.csv
- results/sections/05_feature_sets.md
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from eda_utils import CALENDAR, TABLES, WEATHER_CANDIDATES, ensure_dirs, load_dataset, md_table, write_section

SEEDS = [0, 1, 2, 3, 4]
N_TREES = 300

BASE = ["rain_mm", "rain_hr", "humidity", "sunshine_hr", "wind_ms", "new_snow_cm"]
FEATURE_SETS = {
    "A. 후보 10개 전부": WEATHER_CANDIDATES,
    "B. 평균기온 제외 (9개)": [c for c in WEATHER_CANDIDATES if c != "avg_temp"],
    "C. 전운량 제외 (9개)": [c for c in WEATHER_CANDIDATES if c != "cloud"],
    "D. 평균기온·전운량 제외 (8개, 현재 가안)": ["max_temp", "min_temp"] + BASE,
    "E. 평균기온만 + 전운량 제외 (7개)": ["avg_temp"] + BASE,
    "F. 최고기온만 + 전운량 제외 (7개)": ["max_temp"] + BASE,
    "참고. 날씨 없음 (달력만)": [],
}


def main():
    ensure_dirs()
    df = load_dataset()
    tune = df["tune"].fillna("")
    train, valid = df[tune == "train"].copy(), df[tune == "eval"].copy()

    # 빈칸은 학습 기간 중앙값으로만 채운다 (평가 기간 정보가 섞이지 않게)
    for col in WEATHER_CANDIDATES:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        valid[col] = valid[col].fillna(med)

    rows = []
    for name, weather in FEATURE_SETS.items():
        cols = CALENDAR + weather
        maes = []
        start = time.time()
        for seed in SEEDS:
            model = RandomForestRegressor(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
            model.fit(train[cols], train["rentals"])
            maes.append(mean_absolute_error(valid["rentals"], model.predict(valid[cols])))
        rows.append({
            "조합": name,
            "날씨 변수 수": len(weather),
            "MAE 평균": float(np.mean(maes)),
            "MAE 표준편차": float(np.std(maes)),
            "소요 시간(초)": round(time.time() - start, 1),
        })
        print(f"{name}: MAE {np.mean(maes):,.0f} ± {np.std(maes):,.0f}")

    result = pd.DataFrame(rows).set_index("조합")
    best = result["MAE 평균"].min()
    result["최저 대비 차이(%)"] = (result["MAE 평균"] / best - 1) * 100
    result.to_csv(TABLES / "05_feature_set_mae.csv", encoding="utf-8-sig")

    sets_desc = "\n".join(f"- **{k}**: {', '.join(v) if v else '(없음)'}" for k, v in FEATURE_SETS.items())
    text = f"""
## 5. 날씨 변수 조합별 예측 오차 (조정 단계)

- 학습: 2023년 {len(train):,}일 → 평가: 2024년 {len(valid):,}일
- 모델: 랜덤 포레스트 기본 설정(트리 {N_TREES}개), 난수 시드 {len(SEEDS)}개 반복
- 모든 조합에 달력 변수 6개가 함께 들어간다.
- 설정값 튜닝 전 결과이므로 조합 사이의 상대 비교로만 사용한다.

{md_table(result, 1)}

### 조합 구성

{sets_desc}

> 판단 메모: `eda/DECISIONS.md`에 미리 적어 둔 선택 규칙에 따라 최종 조합을 고르고 그 이유를 5번 항목에 적는다.
"""
    write_section("05_feature_sets", text)
    print(text)


if __name__ == "__main__":
    main()
