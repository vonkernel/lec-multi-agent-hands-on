"""리드: 조사를 계획하고 Sub Agent에 맡긴다"""

import asyncio
import sys
import time

from langchain.agents import create_agent

from researcher import research

PROMPT = """너는 리서치 작업을 감독하는 리드다.
1. 주제를 겹치지 않는 하위 주제 3-4개로 나누되, 핵심 정의·수치·일정을 다루는 하위 주제를 하나 포함한다.
   research는 한 번에 병렬로 호출한다.
2. 조사가 끝나면 하위 주제별로 두세 문장씩 정리해 답한다."""

lead = create_agent("openai:gpt-6-luna", tools=[research], system_prompt=PROMPT, name="lead")


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
    started = time.perf_counter()
    async for update in lead.astream({"messages": [{"role": "user", "content": topic}]}, stream_mode="updates"):
        show(update)
    print(f"총 {time.perf_counter() - started:.0f}초")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
