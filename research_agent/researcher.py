"""researcher: 하위 주제 하나를 검색해 결과를 돌려준다"""

import os
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.mcp import MCPAdapter
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

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


# 6단계: 검색 MCP 서버
SEARCH_SERVER = {"mcpServers": {"search": {
    "command": sys.executable,
    "args": [str(Path(__file__).with_name("search_server.py"))],
    "env": {"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},  # stdio 서버는 환경 변수를 물려받지 않는다
}}}

researcher = None  # 실행을 시작할 때 connect_search가 만든다


async def connect_search():
    """search 서버에서 MCP 도구를 한 번 받아 researcher를 만든다. 서버 프로세스는 실행 내내 재사용된다."""
    global researcher
    async with MCPAdapter(SEARCH_SERVER) as adapter:
        tools = await adapter.list_tools()  # 도구가 연결을 들고 있어, 컨텍스트가 끝난 뒤에도 쓸 수 있다
    researcher = create_agent("openai:gpt-6-luna", tools, system_prompt=PROMPT,
                              response_format=Findings, middleware=[SEARCH_LIMIT], name="researcher")


@tool(args_schema=ResearchTask)
async def research(subject: str, topic: str, scope: str, period: str, must_include: list[str],
                   runtime: ToolRuntime) -> Command:
    """하위 주제 하나를 조사 담당 Agent에게 맡긴다. 여러 하위 주제는 한 번에 병렬로 호출한다."""
    task = (f"전체 주제: {subject}\n하위 주제: {topic}\n범위: {scope}\n기간: {period}\n"
            f"꼭 포함할 것: {', '.join(must_include)}")
    started = time.perf_counter()
    # researcher는 리드의 대화를 보지 않는다. 받는 것은 이 작업 지시 하나뿐이다
    out = await researcher.ainvoke({"messages": [{"role": "user", "content": task}]})
    findings = out["structured_response"]
    used = sum(m.usage_metadata["input_tokens"] for m in out["messages"] if m.type == "ai")  # 세션 3 4단계
    print(f"  [researcher] {time.perf_counter() - started:.0f}초, 입력 {used:,} 토큰 | {topic} ({period})")
    # 세션 4 1단계: researcher가 한 검색 호출과 검색 결과에 나온 URL을 남긴다
    calls = [{"name": c["name"], **c["args"]} for m in out["messages"] if m.type == "ai"
             for c in m.tool_calls]
    urls = sorted({u for m in out["messages"] if m.type == "tool"
                   for u in re.findall(r"https?://[^\s\"'<>)\]]+", m.text)})
    # 4단계: 조사 결과는 state에 두고, 리드에게는 요약만 돌려준다
    return Command(update={
        "findings": [{"topic": topic, **findings.model_dump()}],  # writer가 읽는 조사 결과
        "searches": [{"topic": topic, "calls": calls, "searched_urls": urls}],  # 세션 4 1단계: 분석용 기록
        "messages": [ToolMessage(f"{findings.summary}\n(출처 {len(findings.sources)}개)",
                                 tool_call_id=runtime.tool_call_id)],  # 리드가 보는 것
    })
