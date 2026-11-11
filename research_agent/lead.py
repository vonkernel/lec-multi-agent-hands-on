"""lead: 조사를 계획하고 Sub Agent에 맡긴다"""

import asyncio
import sys

from langchain.agents import create_agent

from llm import model
from researcher import research

PROMPT = """너는 리서치 작업을 감독하는 lead다.
1. 주제를 겹치지 않는 하위 주제 3-4개로 나누되, 핵심 정의·수치·일정을 다루는 하위 주제를 하나 포함한다.
   research는 한 번에 병렬로 호출한다.
   사용자가 정한 기간과 범위 조건은 모든 호출에 그대로 옮긴다.
2. 조사가 끝나면 하위 주제별로 두세 문장씩 정리해 답한다."""

lead = create_agent(model, tools=[research], system_prompt=PROMPT, name="lead")


def show(update):
    """lead의 단계를 한 줄씩 출력한다."""
    for value in update.values():
        for m in value["messages"]:
            if m.type == "ai":
                calls = ", ".join(f"{c['name']}({c['args'].get('topic', '')})" for c in m.tool_calls)
                print(f"[lead] {calls or m.text}")
            elif m.type == "tool":
                print(f"  [{m.name}] {len(m.text):,}자 | {' '.join(m.text.split())[:200]}")


async def main(topic):
    async for update in lead.astream({"messages": [{"role": "user", "content": topic}]},
                                     stream_mode="updates"):
        show(update)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
