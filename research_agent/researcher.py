"""researcher: 하위 주제 하나를 검색해 결과를 돌려준다"""

import time

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool

from tavily_search import fetch, search

PROMPT = """너는 조사 담당이다. 받은 하위 주제 하나만 조사한다.
- search로 2-3번 검색한다. 검색 결과의 본문 일부로 부족할 때만 fetch로 본문을 읽는다.
- 검색 결과에 있는 내용만 쓴다."""

SEARCH_LIMIT = ToolCallLimitMiddleware(tool_name="search", run_limit=30)  # 의도치 않은 무한 검색을 막는 상한


# 2단계: 구조화된 결과
class Source(BaseModel):
    title: str
    url: str
    evidence: str = Field(description="요약의 근거가 되는 문장들. 수치, 날짜, 적용 대상이 든 문장은 모두 그대로 옮긴다")


class Findings(BaseModel):
    summary: str = Field(description="하위 주제에 대한 조사 요약. 수치, 날짜, 적용 대상은 빠짐없이 쓴다")
    sources: list[Source]


researcher = create_agent("openai:gpt-6-luna", tools=[tool(search), tool(fetch)], system_prompt=PROMPT,
                          response_format=Findings, middleware=[SEARCH_LIMIT], name="researcher")


@tool
async def research(topic: str) -> str:
    """하위 주제 하나를 조사 담당 Agent에게 맡긴다. 여러 하위 주제는 한 번에 병렬로 호출한다."""
    started = time.perf_counter()
    # researcher는 리드의 대화를 보지 않는다. 받는 것은 이 하위 주제 하나뿐이다
    out = await researcher.ainvoke({"messages": [{"role": "user", "content": topic}]})
    print(f"  [researcher] {time.perf_counter() - started:.0f}초 | {topic}")
    return out["structured_response"].model_dump_json()  # 요약과 출처를 정해진 형식으로
