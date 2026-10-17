"""리드: 조사를 계획하고 Sub Agent에 맡긴다. 보고서는 직접 쓰지 않는다"""

import asyncio
import operator
import sys
import time
from datetime import datetime
from typing import Annotated, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from checker import check_citations
from researcher import connect_search, research
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


lead = create_agent(
    "openai:gpt-6-luna",
    tools=[research, write_report, check_citations],
    system_prompt=PROMPT,
    state_schema=LeadState,
    name="lead",
    middleware=[
        # 세션 3 1단계: 검색 전에 조사 계획(research 호출)을 사람이 승인한다
        HumanInTheLoopMiddleware(interrupt_on={"research": True}),
    ],
    checkpointer=InMemorySaver(),  # 세션 3 1단계: 멈춘 뒤 재개하려면 실행 상태를 저장해야 한다
)


def show(update):
    """리드의 단계를 한 줄씩 출력한다."""
    for node, value in update.items():
        if node not in ("model", "tools"):  # 미들웨어 노드와 interrupt 알림은 건너뛴다
            continue
        for m in value["messages"]:
            if m.type == "ai":
                calls = ", ".join(f"{c['name']}({c['args'].get('topic', '')})" for c in m.tool_calls)
                print(f"[lead] 입력 {m.usage_metadata['input_tokens']:,} 토큰 → {calls or m.text}")
            elif m.type == "tool":
                print(f"  [{m.name}] {len(m.text):,}자 | {' '.join(m.text.split())[:200]}")


# 세션 3 1단계: 조사 계획 승인
def ask_human(request):
    """research 호출마다 승인, 수정, 거절을 묻는다."""
    decisions = []
    for action in request["action_requests"]:
        args = action["args"]
        print(f"[계획] {args['topic']} | 범위: {args['scope']} | 기간: {args['period']}")
        answer = input("  승인 Enter / 하위 주제 수정 e / 거절 n > ").strip()
        if answer == "e":
            args = {**args, "topic": input("  새 하위 주제 > ")}
            decisions.append({"type": "edit", "edited_action": {"name": "research", "args": args}})
        elif answer == "n":
            decisions.append({"type": "reject", "message": "사람이 이 하위 주제를 계획에서 뺐다"})
        else:
            decisions.append({"type": "approve"})
    return decisions


async def run(inputs, cfg):
    """리드를 실행한다. 승인 단계에서 멈추면 사람에게 묻고 같은 thread로 재개한다."""
    while True:
        async for update in lead.astream(inputs, cfg, stream_mode="updates"):
            show(update)
        state = await lead.aget_state(cfg)
        if not state.interrupts:  # 멈춘 곳이 없으면 끝까지 실행된 것이다
            return state.values
        inputs = Command(resume={"decisions": ask_human(state.interrupts[0].value)})
        # 세션 3 2단계: fork의 cfg에는 갈라진 지점의 checkpoint_id가 있다. 그대로 두면 다음 재개가 그 지점으로 돌아가므로
        # thread만 남겨 최신 지점에서 재개한다
        cfg = {"configurable": {"thread_id": cfg["configurable"]["thread_id"]}}


# 세션 3 2단계: 계획 단계로 돌아가 하위 주제를 바꿔 실행
async def fork(cfg, topics):
    """처음 조사 계획을 세운 checkpoint에서 하위 주제만 바꿔 새 갈래로 실행한다."""
    history = [s async for s in lead.aget_state_history(cfg)]  # 최신 checkpoint가 먼저
    # 모델 호출마다 승인 노드 직전에서 checkpoint가 생긴다. 가장 오래된 것이 첫 계획 직후다
    plan = [s for s in history if s.next == ("HumanInTheLoopMiddleware.after_model",)][-1]
    print(f"checkpoint {len(history)}개 중 step {plan.metadata['step']}(계획 직후)에서 갈라진다")
    old = plan.values["messages"][-1]
    first = old.tool_calls[0]["args"]
    calls = [{"name": "research", "id": f"fork_{i}",
              "args": {**first, "topic": t.strip(), "must_include": []}}  # 전체 주제, 범위, 기간은 처음 계획 그대로
             for i, t in enumerate(topics)]
    new_plan = AIMessage("", tool_calls=calls, id=old.id)  # 같은 id라서 원래 계획 메시지를 교체한다
    # 과거 checkpoint는 그대로 두고, 그 checkpoint를 부모로 하는 새 checkpoint를 만든다
    fork_cfg = await lead.aupdate_state(plan.config, {"messages": [new_plan]})
    return await run(None, fork_cfg)


async def main(topic):
    await connect_search()  # 6단계: search 서버에 한 번 연결한다
    # 실행 ID. checkpoint를 저장하는 thread이자 보고서 폴더 이름
    cfg = {"configurable": {"thread_id": datetime.now().strftime("%m%d-%H%M%S")}}
    started = time.perf_counter()
    await run({"messages": [{"role": "user", "content": topic}]}, cfg)
    print(f"총 {time.perf_counter() - started:.0f}초")

    # 세션 3 2단계
    topics = input("하위 주제를 바꿔 다시 실행하려면 ;로 구분해 입력 (Enter: 끝) > ").strip()
    if topics:
        await fork(cfg, topics.split(";"))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
