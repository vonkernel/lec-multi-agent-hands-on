"""세션 4 2단계: 평가 세트로 Multi Agent를 Phoenix 실험으로 평가한다"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from langgraph.types import Command
from phoenix.client import AsyncClient

from lead import lead

TOPICS = Path("research_agent/topics.jsonl")  # 주제와 보고서에 꼭 들어가야 할 키워드


async def task(input):
    """주제 하나로 lead를 실행하고, 보고서와 lead의 도구 호출 수를 돌려준다."""
    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:6]}"}}
    await lead.ainvoke({"messages": [{"role": "user", "content": input["topic"]}]}, cfg)  # 재작성 질문에서 일시 중지
    state = await lead.ainvoke(Command(resume="n"), cfg)  # 평가에서는 다시 쓰지 않고 끝낸다
    calls = [c for m in state["messages"] if m.type == "ai" for c in m.tool_calls]
    return {"report": Path(state["report"]).read_text(), "steps": len(calls)}


def success(output, expected):
    """필수 키워드가 보고서에 모두 들어 있으면 성공"""
    return all(k in output["report"] for k in expected["keywords"])


def steps(output):
    """lead가 호출한 도구 수"""
    return output["steps"]


async def main():
    rows = [json.loads(line) for line in TOPICS.read_text().splitlines()]
    client = AsyncClient()  # localhost:6006의 Phoenix
    dataset = await client.datasets.create_dataset(
        name=f"research-topics-{datetime.now():%m%d-%H%M%S}",
        inputs=[{"topic": r["topic"]} for r in rows],
        outputs=[{"keywords": r["keywords"]} for r in rows])
    await client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators=[success, steps],
        experiment_name="multi-agent", timeout=600)  # Multi Agent 한 번 실행에 1-2분이 걸린다


if __name__ == "__main__":
    asyncio.run(main())
