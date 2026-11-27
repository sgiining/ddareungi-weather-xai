"""기상청 지상(종관, ASOS) 일자료 조회서비스로 서울(108) 일자료를 받는다.

활용가이드 기준
- 요청 주소: http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList
- 필수 인자: serviceKey, dataCd=ASOS, dateCd=DAY, startDt, endDt(YYYYMMDD, 전일까지), stnIds
- 선택 인자: numOfRows, pageNo, dataType(XML/JSON)
- 최대 메시지 크기 1000KB → 1년씩 나누고, 페이지를 넘겨 가며 받는다.

출력
- data/raw/asos_seoul_daily_raw.csv      : API가 준 모든 열
- data/raw/asos_seoul_daily_selected.csv : 연구에 쓸 열만 (숫자로 변환)
"""

from datetime import date
from pathlib import Path

import pandas as pd

from api_common import call_api, check_header, extract_items, get_key, total_count

URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STATION = "108"  # 서울
START = date(2023, 1, 1)
END = date(2026, 6, 30)
ROWS_PER_PAGE = 200

# 연구에 쓸 열 (최종정리본 3.2절)
SELECTED = {
    "tm": "date",
    "avgTa": "avg_temp",       # 평균 기온(°C)
    "maxTa": "max_temp",       # 최고 기온(°C)
    "minTa": "min_temp",       # 최저 기온(°C)
    "sumRn": "rain_mm",        # 일강수량(mm)
    "sumRnDur": "rain_hr",     # 강수 계속시간(hr)
    "avgWs": "wind_ms",        # 평균 풍속(m/s)
    "avgRhm": "humidity",      # 평균 상대습도(%)
    "sumSsHr": "sunshine_hr",  # 합계 일조시간(hr)
    "ddMefs": "new_snow_cm",   # 일 최심신적설(cm)
    "avgTca": "cloud",         # 평균 전운량(10분위)
}

OUT_DIR = Path("data/raw")


def year_ranges(start: date, end: date):
    """[start, end]를 1년 단위 구간으로 나눈다."""
    for y in range(start.year, end.year + 1):
        s = max(start, date(y, 1, 1))
        e = min(end, date(y, 12, 31))
        yield s, e


def fetch_range(key: str, s: date, e: date) -> list:
    rows, page = [], 1
    while True:
        params = {
            "serviceKey": key,
            "numOfRows": ROWS_PER_PAGE,
            "pageNo": page,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": s.strftime("%Y%m%d"),
            "endDt": e.strftime("%Y%m%d"),
            "stnIds": STATION,
        }
        data = call_api(URL, params)
        check_header(data)
        items = extract_items(data)
        rows.extend(items)
        total = total_count(data)
        print(f"  {s}~{e} page {page}: {len(items)}건 (누적 {len(rows)}/{total})")
        if not items or len(rows) >= total:
            break
        page += 1
    return rows


def main():
    key = get_key("ASOS_API_KEY")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for s, e in year_ranges(START, END):
        print(f"[ASOS] {s} ~ {e} 요청")
        all_rows.extend(fetch_range(key, s, e))

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT_DIR / "asos_seoul_daily_raw.csv", index=False, encoding="utf-8-sig")

    # 필요한 열만 남기고 숫자로 변환 (빈 문자열은 NaN이 된다)
    missing_cols = [c for c in SELECTED if c not in raw.columns]
    if missing_cols:
        print(f"주의: 응답에 없는 열 {missing_cols}")
    sel = raw[[c for c in SELECTED if c in raw.columns]].rename(columns=SELECTED)
    sel["date"] = pd.to_datetime(sel["date"])
    for col in sel.columns.drop("date"):
        sel[col] = pd.to_numeric(sel[col], errors="coerce")
    sel = sel.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    sel.to_csv(OUT_DIR / "asos_seoul_daily_selected.csv", index=False, encoding="utf-8-sig")

    # ---- 점검 출력 ----
    expected = (END - START).days + 1
    print("\n[점검]")
    print(f"행 수: {len(sel)} (기대값 {expected})")
    full = pd.date_range(START, END, freq="D")
    lost = full.difference(sel["date"])
    print(f"빠진 날짜: {len(lost)}개 {list(lost[:10].date)}")
    print("열별 결측(NaN) 수:")
    print(sel.isna().sum().to_string())
    print("\n※ rain_mm, new_snow_cm의 NaN이 '비·눈 없음'을 뜻하는지 확인한 뒤 전처리에서 0으로 채운다.")
    print("  (예: rain_hr도 NaN이고 rain_mm이 NaN인 날이 모두 맑은 날인지 확인)")


if __name__ == "__main__":
    main()
