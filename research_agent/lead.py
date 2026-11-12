"""lead: 조사를 계획하고 Sub Agent에 맡긴다. 보고서는 직접 쓰지 않는다"""

import asyncio
import operator
import sys
from datetime import datetime
from typing import Annotated, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from llm import model
from researcher import research
from writer import write_report

# 세션 4 1단계: LangChain 실행을 trace로 기록해 Phoenix(localhost:6006)로 보낸다
tracer_provider = register(project_name="research-agent", protocol="http/protobuf", batch=True)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

PROMPT = """너는 리서치 보고서 작업을 감독하는 lead다. 보고서 본문은 직접 쓰지 않는다.
1. 주제를 겹치지 않는 하위 주제 3-4개로 나누되, 핵심 정의·수치·일정을 다루는 하위 주제를 하나 포함한다.
   research는 한 번에 병렬로 호출한다.
   사용자가 정한 기간과 범위 조건은 모든 호출에 그대로 옮긴다.
2. 조사가 끝나면 write_report로 보고서를 쓰게 한다.
3. 끝나면 보고서 경로를 한 문장으로 답한다."""


class LeadState(AgentState):
    findings: NotRequired[Annotated[list[dict], operator.add]]  # researcher가 남긴 조사 결과
    report: NotRequired[str]                                    # writer가 저장한 보고서 경로


@after_agent
def ask_rewrite(state, runtime):
    """세션 3: 보고서를 다 쓰면 일시 중지하고, 다시 쓸지 사람에게 묻는다."""
    interrupt(f"보고서를 다시 쓸까요? ({state['report']})")  # 답은 main이 재개 값으로 받아 쓴다
    return {"messages": []}  # 아무것도 반환하지 않으면 업데이트가 None이 되어 show가 읽지 못한다


lead = create_agent(model, tools=[research, write_report], system_prompt=PROMPT,
                    state_schema=LeadState, name="lead", middleware=[ask_rewrite],
                    checkpointer=InMemorySaver())  # 세션 3: 단계마다 state를 저장한다


def show(update):
    """lead의 단계를 한 줄씩 출력한다."""
    for value in update.values():
        for m in value["messages"]:
            if m.type == "ai":
                calls = ", ".join(f"{c['name']}({c['args'].get('topic', '')})" for c in m.tool_calls)
                print(f"[lead] {calls or m.text}")
            elif m.type == "tool":
                print(f"  [{m.name}] {len(m.text):,}자 | {' '.join(m.text.split())[:200]}")


async def run(inp, cfg):
    """lead를 실행하고, 일시 중지되면 사람에게 보여 줄 질문을 돌려준다."""
    question = None
    async for update in lead.astream(inp, cfg, stream_mode="updates"):  # 스트림은 끝까지 읽는다
        if "__interrupt__" in update:
            question = update["__interrupt__"][0].value
        else:
            show(update)
    return question


def rewind_point(cfg, answer):
    """세션 3: r은 사용자 입력 직후, w는 write_report 실행 직전 checkpoint를 찾는다."""
    for snap in lead.get_state_history(cfg):  # 최신 checkpoint부터
        messages = snap.values.get("messages", [])
        if answer == "r" and snap.next == ("model",) and len(messages) == 1:
            return snap.config  # 조사 결과(findings)가 아직 없는 지점
        if answer == "w" and snap.next == ("tools",) and any(
                c["name"] == "write_report" for c in messages[-1].tool_calls):
            return snap.config  # 조사 결과는 state에 있고 보고서는 아직 쓰기 전인 지점


async def main(topic):
    cfg = {"configurable": {"thread_id": datetime.now().strftime("%m%d-%H%M%S")}}  # 실행 ID. 보고서 폴더 이름
    question = await run({"messages": [{"role": "user", "content": topic}]}, cfg)
    while question:
        answer = input(f"\n{question} [n] 끝 / [r] 리서치부터 / [w] 조사 결과로 다시 쓰기 > ").strip()
        await run(Command(resume=answer), cfg)  # 같은 thread로 재개하면 ask_rewrite가 이 값을 받는다
        if answer not in ("r", "w"):
            break
        # 세션 3: 고른 checkpoint부터 다시 실행한다. 이후 checkpoint는 새 갈래로 쌓인다
        question = await run(None, rewind_point(cfg, answer))
    print(f"최종 보고서: {lead.get_state(cfg).values['report']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
