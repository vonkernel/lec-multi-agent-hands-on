"""리드: 조사를 계획하고 Sub Agent에 맡긴다. 보고서는 직접 쓰지 않는다"""

import asyncio
import operator
import sys
import time
from datetime import datetime
from typing import Annotated, NotRequired

from langchain.agents import AgentState, create_agent

from checker import check_citations
from researcher import research
from writer import write_report

PROMPT = """너는 리서치 보고서 작업을 감독하는 리드다. 보고서 본문은 직접 쓰지 않는다.
1. 주제를 겹치지 않는 하위 주제 3-4개로 나누되, 핵심 정의·수치·일정을 다루는 하위 주제를 하나 포함한다.
   research는 한 번에 병렬로 호출한다.
   사용자가 정한 기간과 범위 조건은 모든 호출에 그대로 옮긴다.
2. 조사가 끝나면 write_report로 보고서를 쓰게 한다.
3. check_citations로 검토한다. '수정 필요'면 지적 사항을 write_report의 instructions로 넘겨 고치게 하고 다시 검토한다. 수정 지시는 최대 3번이다.
4. 끝나면 최종 보고서 경로와 검토 결과를 두세 문장으로 답한다."""


class LeadState(AgentState):
    findings: NotRequired[Annotated[list[dict], operator.add]]  # researcher가 남긴 조사 결과
    drafts: NotRequired[Annotated[list[str], operator.add]]     # writer가 저장한 보고서 경로


lead = create_agent("openai:gpt-6-luna", tools=[research, write_report, check_citations], system_prompt=PROMPT,
                    state_schema=LeadState, name="lead")


def show(update):
    """리드의 단계를 한 줄씩 출력한다."""
    for value in update.values():
        for m in value["messages"]:
            if m.type == "ai":
                calls = ", ".join(f"{c['name']}({c['args'].get('topic', '')})" for c in m.tool_calls)
                print(f"[lead] 입력 {m.usage_metadata['input_tokens']:,} 토큰 → {calls or m.text}")
            elif m.type == "tool":
                print(f"  [{m.name}] {len(m.text):,}자 | {' '.join(m.text.split())[:200]}")


async def main(topic):
    cfg = {"configurable": {"thread_id": datetime.now().strftime("%m%d-%H%M%S")}}  # 실행 ID. 보고서 폴더 이름
    started = time.perf_counter()
    async for update in lead.astream({"messages": [{"role": "user", "content": topic}]}, cfg,
                                     stream_mode="updates"):
        show(update)
    print(f"총 {time.perf_counter() - started:.0f}초")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
