# ddareungi-weather-xai

기상 조건에 따른 서울 공공자전거(따릉이) 일별 대여량 예측 모델의 설명 분석
(2026년도 2학기 설명가능한 인공지능(XAI) Mini Research Project)

달력 정보만 쓴 랜덤 포레스트와 날씨 정보를 더한 랜덤 포레스트의 예측 오차를 비교하고,
SHAP(SHapley Additive exPlanations)과 기온 반응 곡선(ICE 응용)으로 모델이 날씨를 어떻게 반영했는지 분석합니다.

## 자료

| 자료 | 제공 기관 | 기간·단위 | 받는 방법 |
|---|---|---|---|
| 따릉이 일별 대여건수 | 서울 열린데이터광장 (OA-14994) | 2023-01-01 ~ 2026-06-30, 일 단위, 반기별 파일 7개 | 웹사이트에서 내려받아 `data/`에 넣음 |
| ASOS 일자료 | 기상청 (공공데이터포털 15059093) | 같은 기간, 서울 지점(108) | `api_code/fetch_asos.py` |
| 특일 정보 | 한국천문연구원 (공공데이터포털 15012690) | 2023 ~ 2026년 6월 | `api_code/fetch_holidays.py` |

`data/`와 `.env`(인증키)는 저장소에 올리지 않습니다(`.gitignore`).

## 폴더 구조

```
ddareungi-weather-xai/
├─ api_code/            기상·공휴일 API 수집
├─ build_dataset.py     세 자료를 날짜로 합쳐 data/processed/dataset.csv 생성 (최종 변수 목록 정의)
├─ eda/                 탐색 분석, 변수 선택, 결정 기록
│  ├─ DECISIONS.md      결과를 보기 전에 정한 규칙과 판단 근거 (1~8번)
│  └─ results/          EDA 표·그림·보고서
├─ experiments/         모델 학습·평가·설명 분석 (실행 순서는 experiments/README.md)
├─ results/             실험 결과
│  ├─ tables/           CSV, JSON
│  └─ figures/          PNG
└─ requirements.txt
```

## 실행 환경

| 항목 | 버전 |
|---|---|
| Python | 3.12.3 |
| 라이브러리 | `requirements.txt` 참고 (pandas, numpy, scikit-learn, matplotlib, shap 등) |
| 운영체제·CPU | Windows 10, Intel64 Family 6 Model 142, GPU 사용하지 않음 |

```
py -m pip install -r requirements.txt
```

## 재현 순서

모든 명령은 프로젝트 폴더에서 실행합니다.

1. **인증키 준비:** 프로젝트 폴더에 `.env` 파일을 만들고 공공데이터포털 인증키를 넣습니다.
   ```
   ASOS_API_KEY=발급받은_인증키
   HOLIDAY_API_KEY=발급받은_인증키
   ```
2. **자료 수집과 결합**
   ```
   py api_code/fetch_asos.py
   py api_code/fetch_holidays.py
   py build_dataset.py
   ```
   따릉이 반기별 파일(`*일별 대여건수*.csv` / `.xlsx`)은 미리 `data/`에 넣어 둡니다.
3. **탐색 분석 (변수 선택 근거)**
   ```
   py eda/run_all.py
   ```
4. **실험**
   ```
   py experiments/01_tune.py        # 설정값 탐색 (2023 학습 → 2024 평가)
   py experiments/02_main.py        # 4개 모델 × 테스트 2기간 성능
   py experiments/03_ablation.py    # 날씨 묶음 제거 + 직전 12개월 학습
   py experiments/04_shap.py        # SHAP, 기온 반응 곡선, 오차 큰 날
   py experiments/05_stability.py   # 설명 안정성 (Jaccard)
   ```
   각 스크립트는 `--quick` 옵션으로 코드만 빠르게 점검할 수 있습니다. 결과 파일 이름 끝에 `_quick`이 붙습니다.

## 평가 설계

시간 순서를 지켜 평가하고, 테스트 결과를 보고 입력이나 설정을 바꾸지 않습니다.

| 단계 | 학습 | 평가 | 용도 |
|---|---|---|---|
| 조정 | 2023년 | 2024년 | 변수·설정 선택 |
| 테스트 2025 | 2023~2024년 | 2025년 | 성능 비교, 설명 분석 |
| 테스트 2026H1 | 2023~2025년 | 2026년 1~6월 | 성능 비교, 설명 안정성 |
| 보조 | 각 테스트 직전 12개월 | 위와 같음 | 학습 기간 길이 비교 |

- **입력:** 달력 변수 6개(월·요일은 원-핫 인코딩), 날씨 변수 9개(기온·강수·습도·하늘·바람·눈의 5개 묶음)
- **모델:** 월·요일 평균, 선형회귀, 달력 랜덤 포레스트, 기상 추가 랜덤 포레스트
- **지표:** MAE, RMSE, R², ME. 랜덤 포레스트는 시드 5개의 평균 ± 표준편차로 보고합니다.

## 결과 파일

| 내용 | 파일 (`results/tables/`) |
|---|---|
| 선택된 설정 | `01_tune_selected.json` |
| 모델별 성능 | `02_main_table10.csv`, `02_main_summary.csv` |
| 날씨 묶음 제거 실험 | `03_ablation_table11.csv` |
| 직전 12개월 학습 비교 | `03_last12_table.csv` |
| SHAP 중요도·조건별 기여도 | `04_importance.csv`, `04_by_segment.csv` |
| 오차 큰 날 | `04_large_error_days_*.csv` |
| 설명 안정성 | `05_stability_table.csv` |

그림은 `results/figures/04_*.png`에 있습니다.

## 주요 결과 요약

- 날씨 정보를 더하면 달력 랜덤 포레스트보다 MAE가 2025년 41.9%, 2026년 상반기 48.4% 줄었습니다.
- 날씨 묶음 제거 실험과 SHAP 묶음 중요도가 모두 기온 > 강수 > 습도·하늘 > 눈·바람 순서였습니다.
- 모든 모델이 2025년을 과대예측했습니다(ME가 음수).
  - 직전 12개월 학습과 연도의 SHAP 값으로 원인을 확인했습니다.
  - 2025년은 학습 자료에 없던 대여 수준 하락이 주된 원인으로 보입니다.
  - 2026년 상반기는 과거의 높은 수준이 학습에 섞인 영향이 컸습니다.
- 상위 날씨 묶음 3개는 시드 간, 기간 간 Jaccard 지수가 모두 1.0이었습니다. 상위 변수 5개는 기간 간 0.667로, 5위 한 자리만 달랐습니다.

설명 결과는 모델이 학습한 예측 근거이며, 날씨가 대여량에 미치는 인과적 효과를 뜻하지 않습니다.
