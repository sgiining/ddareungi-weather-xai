"""한국천문연구원 특일 정보(공공데이터포털 15012690)로 공휴일 목록을 받는다.

- 서비스: SpcdeInfoService, 기능: getRestDeInfo (공휴일 정보조회)
- 인자: ServiceKey, solYear(연), solMonth(월, 2자리), numOfRows, _type=json
- 응답 항목: locdate(YYYYMMDD), dateName, isHoliday(Y/N), dateKind, seq
- 월 단위로 조회 → 2023-01 ~ 2026-06 = 42회 호출

출력: data/raw/holidays_kr.csv
"""

from pathlib import Path

import pandas as pd

from api_common import call_api, check_header, extract_items, get_key

URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
START_YM = (2023, 1)
END_YM = (2026, 6)
OUT_DIR = Path("data/raw")


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    key = get_key("HOLIDAY_API_KEY")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for y, m in months(START_YM, END_YM):
        params = {
            "ServiceKey": key,
            "solYear": str(y),
            "solMonth": f"{m:02d}",
            "numOfRows": 50,
            "_type": "json",
        }
        data = call_api(URL, params)
        check_header(data)
        items = extract_items(data)
        print(f"[공휴일] {y}-{m:02d}: {len(items)}건")
        rows.extend(items)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["locdate"].astype(str), format="%Y%m%d")
    df = df[["date", "dateName", "isHoliday", "dateKind"]].sort_values("date")
    df = df.drop_duplicates("date").reset_index(drop=True)
    df.to_csv(OUT_DIR / "holidays_kr.csv", index=False, encoding="utf-8-sig")

    # ---- 점검 출력: 대체공휴일·선거일이 들어 있는지 눈으로 확인 ----
    print(f"\n총 {len(df)}일 (isHoliday=Y: {(df['isHoliday'] == 'Y').sum()}일)")
    print(df.to_string(index=False))
    print("\n※ 대체공휴일·선거일(예: 2026-06-03 지방선거)이 목록에 있는지 확인한다.")
    print("  빠져 있으면 data/raw/holidays_manual.csv에 직접 추가하고, 그 사실을 계획서 전처리 절에 적는다.")


if __name__ == "__main__":
    main()
