"""Tavily 검색 함수. 세션 1에서는 LangChain 도구로, 세션 2에서는 MCP 서버 도구로 감싼다."""

import json
import os
from typing import Literal

from tavily import TavilyClient

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search(query: str, lang: Literal["ko", "en"] = "ko") -> str:
    """웹을 검색해 상위 5개 결과의 제목, URL, 본문 일부를 돌려준다.
    한국 관련 주제는 lang="ko", 해외 동향은 lang="en"으로 검색한다."""
    korean = {"language": "ko", "country": "south korea"} if lang == "ko" else {}
    response = client.search(query, search_depth="basic", max_results=5, **korean)  # basic: 1크레딧
    return json.dumps(response, ensure_ascii=False)


def fetch(url: str, query: str) -> str:
    """URL 하나에서 query와 관련된 본문 부분만 가져온다. 검색 결과의 본문 일부로 부족할 때만 쓴다."""
    response = client.extract(url, query=query, chunks_per_source=3)  # 관련도가 높은 조각 3개만
    return json.dumps(response, ensure_ascii=False)


if __name__ == "__main__":
    query = "주 52시간 근무제 시행 시기"
    found = search(query)
    print(f"search: {len(found):,}자")
    response = json.loads(found)
    print("응답 필드:", list(response))
    print("결과 필드:", list(response["results"][0]))
    page = fetch(response["results"][0]["url"], query)
    print(f"fetch: {len(page):,}자")
