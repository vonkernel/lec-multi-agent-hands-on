"""Tavily 검색 함수. 세션 1에서는 LangChain 도구로, 세션 2에서는 MCP 서버 도구로 감싼다."""

import json
import os
from typing import Literal

from tavily import TavilyClient

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search(query: str, lang: Literal["ko", "en"] = "ko", start_date: str | None = None,
           end_date: str | None = None) -> str:
    """웹을 검색해 상위 5개 결과의 제목, URL, 본문 일부를 돌려준다.
    한국 관련 주제는 lang="ko", 해외 동향은 lang="en"으로 검색한다.
    기간 조건이 있으면 start_date, end_date(YYYY-MM-DD)를 주어 그 기간에 발행된 문서만 찾는다."""
    korean = {"language": "ko", "country": "south korea"} if lang == "ko" else {}
    # 세션 2 3단계: 기간이 있으면 발행일로 거르고, 발행일을 알 수 없는 문서도 뺀다
    period = ({"start_date": start_date, "end_date": end_date, "filter_by_published_date": True}
              if start_date or end_date else {})
    response = client.search(query, search_depth="basic", max_results=5, **korean, **period)  # basic: 1크레딧
    return json.dumps(response, ensure_ascii=False)


def fetch(url: str) -> str:
    """URL 하나의 본문을 가져온다. 검색 결과의 본문 일부로 부족할 때만 쓴다."""
    response = client.extract(url)
    return json.dumps(response, ensure_ascii=False)


if __name__ == "__main__":
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")  # 토큰 수 근사치
    found = search("주 52시간 근무제 시행 시기")
    print("search:", len(enc.encode(found)), "토큰")
    response = json.loads(found)
    print("응답 필드:", list(response))
    print("결과 필드:", list(response["results"][0]))
    page = fetch(response["results"][0]["url"])
    print("fetch:", len(enc.encode(page)), "토큰")
