"""researcher: 하위 주제 하나를 검색해 결과를 돌려준다"""

import time

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool

from tavily_search import fetch, search

PROMPT = """너는 조사 담당이다. 받은 하위 주제 하나만 조사한다.
- search로 2-3번 검색한다. 검색 결과의 본문 일부로 부족할 때만 fetch로 본문을 읽는다.
- 검색어에는 하위 주제와 함께 전체 주제의 핵심어, 범위, 꼭 포함할 것을 넣는다.
- 기간은 search의 start_date, end_date로 넘긴다. 끝이 없는 기간이면 end_date를 비운다.
- 범위와 기간 조건을 지키고, 검색 결과에 있는 내용만 쓴다."""

SEARCH_LIMIT = ToolCallLimitMiddleware(tool_name="search", run_limit=30)  # 의도치 않은 무한 검색을 막는 상한


# 2단계: 구조화된 결과
class Source(BaseModel):
    title: str
    url: str
    evidence: str = Field(description="요약의 근거가 되는 문장들. 수치, 날짜, 적용 대상이 든 문장은 모두 그대로 옮긴다")


class Findings(BaseModel):
    summary: str = Field(description="하위 주제에 대한 조사 요약. 수치, 날짜, 적용 대상은 빠짐없이 쓴다")
    sources: list[Source]


# 3단계: 위임 인자 구조화
class ResearchTask(BaseModel):
    subject: str = Field(description="사용자가 요청한 전체 주제. 그대로 옮긴다")
    topic: str = Field(description="조사할 하위 주제 하나")
    scope: str = Field(description="조사 범위(지역, 대상 등). 사용자가 정한 조건을 그대로 옮긴다")
    period: str = Field(description="조사 기간. 사용자가 정한 기간을 그대로 옮긴다. 없으면 '제한 없음'")
    must_include: list[str] = Field(description="결과에 꼭 들어가야 할 항목")


researcher = create_agent("openai:gpt-6-luna", tools=[tool(search), tool(fetch)], system_prompt=PROMPT,
                          response_format=Findings, middleware=[SEARCH_LIMIT], name="researcher")


@tool(args_schema=ResearchTask)
async def research(subject: str, topic: str, scope: str, period: str, must_include: list[str]) -> str:
    """하위 주제 하나를 조사 담당 Agent에게 맡긴다. 여러 하위 주제는 한 번에 병렬로 호출한다."""
    task = (f"전체 주제: {subject}\n하위 주제: {topic}\n범위: {scope}\n기간: {period}\n"
            f"꼭 포함할 것: {', '.join(must_include)}")
    started = time.perf_counter()
    # researcher는 리드의 대화를 보지 않는다. 받는 것은 이 작업 지시 하나뿐이다
    out = await researcher.ainvoke({"messages": [{"role": "user", "content": task}]})
    print(f"  [researcher] {time.perf_counter() - started:.0f}초 | {topic} ({period})")
    return out["structured_response"].model_dump_json()  # 요약과 출처를 정해진 형식으로
