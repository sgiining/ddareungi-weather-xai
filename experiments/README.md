# experiments/ — 모델 학습·평가·설명 분석

EDA(`eda/`)에서 정한 변수와 `eda/DECISIONS.md` 7·8번 규칙에 따라 실험합니다.
결과는 프로젝트 폴더의 `results/`(tables, figures)에 저장됩니다.

## 실행 순서

프로젝트 폴더(`D:\xai\ddareungi-weather-xai`)에서 실행합니다. 먼저 `py build_dataset.py`로 `data/processed/dataset.csv`가 있어야 합니다.

```
py -m pip install -r requirements.txt
py experiments/01_tune.py --quick    # 코드 점검 (1분 이내)
py experiments/01_tune.py            # 설정값 탐색 본 실행
```

| 번호 | 파일 | 하는 일 | 계획서 |
|---|---|---|---|
| — | `common.py` | 데이터 분할, 빈칸 채우기, 원-핫 입력, SHAP 합산 (공통 함수) | — |
| 1 | `01_tune.py` | 조정 단계 설정값 탐색 (108개 조합 × 시드 5개, B1·P) | 표 7 |
| — | `analysis_rules.py` | 오차가 큰 날짜 선정, 기온 반응 곡선 계산 (5단계에서 사용) | 4.5.3 |

## 결과 파일

```
results/tables/
├─ 01_tune_B1.csv, 01_tune_P.csv   ← 조합별 MAE 평균·표준편차
└─ 01_tune_selected.json            ← 모델별 선택 설정 (이후 실험이 읽음)
```

`--quick`으로 실행한 결과는 파일 이름 끝에 `_quick`이 붙어 본 실행 결과와 섞이지 않습니다.
