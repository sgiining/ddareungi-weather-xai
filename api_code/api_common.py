"""공공데이터포털 API 공통 함수 (ASOS, 특일 정보에서 함께 사용)."""

import os
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()  # 프로젝트 폴더의 .env 파일에서 인증키를 읽는다


def get_key(name: str) -> str:
    """환경 변수(.env)에서 인증키를 읽는다. 없으면 바로 알려 준다."""
    key = os.getenv(name)
    if not key:
        raise RuntimeError(f".env 파일에 {name} 값이 없습니다.")
    return key.strip()


def call_api(url: str, params: dict, retries: int = 3, wait: float = 1.0) -> dict:
    """GET 요청을 보내고 응답을 dict로 돌려준다.

    - JSON으로 오면 그대로, XML로 오면(인증 오류 등) dict로 바꿔 돌려준다.
    - 네트워크 오류는 retries번까지 다시 시도한다.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            text = resp.text.strip()
            if text.startswith("{"):
                return resp.json()
            return _xml_to_dict(text)
        except (requests.RequestException, ValueError, ET.ParseError) as e:
            last_error = e
            print(f"  요청 실패({attempt}/{retries}): {e}")
            time.sleep(wait * attempt)
    raise RuntimeError(f"API 요청이 계속 실패했습니다: {last_error}")


def _xml_to_dict(text: str) -> dict:
    """XML 응답을 JSON 응답과 같은 모양(response/header, response/body)으로 바꾼다."""
    root = ET.fromstring(text)

    # 인증키 오류는 <OpenAPI_ServiceResponse> 형태로 올 수 있다
    if root.tag == "OpenAPI_ServiceResponse":
        header = root.find("cmmMsgHeader")
        msg = header.findtext("returnAuthMsg") if header is not None else text[:200]
        code = header.findtext("returnReasonCode") if header is not None else "?"
        return {"response": {"header": {"resultCode": code, "resultMsg": msg}, "body": {}}}

    header = root.find("header")
    body = root.find("body")
    items = []
    if body is not None and body.find("items") is not None:
        for item in body.find("items").findall("item"):
            items.append({child.tag: child.text for child in item})
    return {
        "response": {
            "header": {
                "resultCode": header.findtext("resultCode") if header is not None else "?",
                "resultMsg": header.findtext("resultMsg") if header is not None else "",
            },
            "body": {
                "items": {"item": items},
                "totalCount": body.findtext("totalCount") if body is not None else "0",
            },
        }
    }


def check_header(data: dict) -> None:
    """resultCode가 정상(0 또는 00)이 아니면 멈춘다."""
    header = data.get("response", {}).get("header", {})
    code = str(header.get("resultCode", "")).strip()
    if code not in ("0", "00"):
        raise RuntimeError(f"API 오류 resultCode={code}, resultMsg={header.get('resultMsg')}")


def extract_items(data: dict) -> list:
    """body.items.item을 항상 list로 돌려준다 (결과가 1건이면 dict, 0건이면 빈 값으로 오는 경우 처리)."""
    body = data.get("response", {}).get("body", {}) or {}
    items = body.get("items")
    if not items:  # 결과 0건이면 "" 또는 None으로 온다
        return []
    item = items.get("item", [])
    if isinstance(item, dict):
        return [item]
    return list(item)


def total_count(data: dict) -> int:
    body = data.get("response", {}).get("body", {}) or {}
    try:
        return int(body.get("totalCount", 0))
    except (TypeError, ValueError):
        return 0
