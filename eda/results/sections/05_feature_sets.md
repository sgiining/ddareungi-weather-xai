## 5. 날씨 변수 조합별 예측 오차 (조정 단계)

- 학습: 2023년 365일 → 평가: 2024년 366일
- 모델: 랜덤 포레스트 기본 설정(트리 300개), 난수 시드 5개 반복
- 모든 조합에 달력 변수 6개가 함께 들어간다.
- 설정값 튜닝 전 결과이므로 조합 사이의 상대 비교로만 사용한다.

| 조합 | 날씨 변수 수 | MAE 평균 | MAE 표준편차 | 소요 시간(초) | 최저 대비 차이(%) |
| --- | --- | --- | --- | --- | --- |
| A. 후보 10개 전부 | 10 | 11,077.2 | 33.9 | 2.3 | 0.0 |
| B. 평균기온 제외 (9개) | 9 | 11,075.7 | 35.6 | 2.5 | 0.0 |
| C. 전운량 제외 (9개) | 9 | 11,173.6 | 33.4 | 2.2 | 0.9 |
| D. 평균기온·전운량 제외 (8개, 현재 가안) | 8 | 11,236.6 | 29.2 | 2.1 | 1.5 |
| E. 평균기온만 + 전운량 제외 (7개) | 7 | 11,975.9 | 23.8 | 2.1 | 8.1 |
| F. 최고기온만 + 전운량 제외 (7개) | 7 | 11,969.1 | 52.0 | 2.1 | 8.1 |
| 참고. 날씨 없음 (달력만) | 0 | 25,398.4 | 79.5 | 2.0 | 129.3 |

### 조합 구성

- **A. 후보 10개 전부**: avg_temp, max_temp, min_temp, rain_mm, rain_hr, humidity, sunshine_hr, cloud, wind_ms, new_snow_cm
- **B. 평균기온 제외 (9개)**: max_temp, min_temp, rain_mm, rain_hr, humidity, sunshine_hr, cloud, wind_ms, new_snow_cm
- **C. 전운량 제외 (9개)**: avg_temp, max_temp, min_temp, rain_mm, rain_hr, humidity, sunshine_hr, wind_ms, new_snow_cm
- **D. 평균기온·전운량 제외 (8개, 현재 가안)**: max_temp, min_temp, rain_mm, rain_hr, humidity, sunshine_hr, wind_ms, new_snow_cm
- **E. 평균기온만 + 전운량 제외 (7개)**: avg_temp, rain_mm, rain_hr, humidity, sunshine_hr, wind_ms, new_snow_cm
- **F. 최고기온만 + 전운량 제외 (7개)**: max_temp, rain_mm, rain_hr, humidity, sunshine_hr, wind_ms, new_snow_cm
- **참고. 날씨 없음 (달력만)**: (없음)

> 판단 메모: `eda/DECISIONS.md`에 미리 적어 둔 선택 규칙에 따라 최종 조합을 고르고 그 이유를 5번 항목에 적는다.
