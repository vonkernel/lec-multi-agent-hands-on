"""세션 4 3단계: Phoenix 실험으로 단일 Agent와 Multi Agent를 같은 평가 세트에서 비교한다"""

import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from phoenix.client import AsyncClient

from lead import run
from researcher import connect_search
from single_agent import agent as single_agent

PRICE = {"gpt-6-luna": (0.10, 0.50)}  # USD / 100만 토큰 (입력, 출력)
URL = re.compile(r"https?://[^\s\"'<>)\]]+")


def cost(usage):
    """모델별 토큰 사용량을 비용(USD)으로 바꾼다."""
    total = 0.0
    for model, u in usage.items():
        # 응답의 모델 이름에는 날짜가 붙는다(gpt-6-luna-2026-...). 앞부분으로 단가를 찾는다
        price_in, price_out = next(price for name, price in PRICE.items() if model.startswith(name))
        total += (u["input_tokens"] * price_in + u["output_tokens"] * price_out) / 1e6
    return total


# 실행: 구성마다 보고서, 단계 수, 검색 수, 검색 결과에 나온 URL을 돌려준다
def tool_calls(messages):
    """AI 메시지가 부른 도구 호출을 모두 모은다."""
    calls = []
    for m in messages:
        if m.type == "ai":
            calls.extend(m.tool_calls)
    return calls


def urls_in_tool_results(messages):
    """도구 결과 메시지에 나온 URL을 모두 모은다."""
    urls = set()
    for m in messages:
        if m.type == "tool":
            urls.update(URL.findall(m.text))
    return urls


async def run_single(topic):
    out = await single_agent.ainvoke({"messages": [{"role": "user", "content": topic}]})
    calls = tool_calls(out["messages"])
    return {"report": out["messages"][-1].text,
            "steps": len(calls),     # 단일 Agent는 모든 단계가 검색이다
            "searches": len(calls),
            "searched": urls_in_tool_results(out["messages"])}


def approve_all(request):
    """조사 계획을 모두 승인한다. 두 구성을 같은 조건으로 비교하려고 사람 대신 쓴다."""
    return [{"type": "approve"}] * len(request["action_requests"])


async def run_multi(topic):
    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:6]}"}}
    state = await run({"messages": [{"role": "user", "content": topic}]}, cfg, ask=approve_all)
    lead_calls = tool_calls(state["messages"])
    search_calls = [c for s in state["searches"] for c in s["calls"]]
    return {"report": Path(state["drafts"][-1]).read_text(),
            "steps": len(lead_calls) + len(search_calls),  # 리드의 위임 + researcher의 검색
            "searches": len(search_calls),
            "searched": {u for s in state["searches"] for u in s["searched_urls"]}}


# task: Phoenix가 데이터셋의 주제마다 부른다. 주제(input)를 받아 평가기가 볼 결과(output)를 돌려준다
async def measure(runner, topic):
    """구성 하나를 실행하고 보고서와 함께 단계, 검색, 비용, 시간을 잰다."""
    started = time.perf_counter()
    with get_usage_metadata_callback() as cb:  # 이 실행 안의 모든 모델 호출 토큰을 모델별로 합산
        result = await runner(topic)
    return {"report": result["report"], "searched": sorted(result["searched"]),
            "steps": result["steps"], "searches": result["searches"],
            "cost": cost(cb.usage_metadata), "sec": time.perf_counter() - started}


async def single_task(input):
    return await measure(run_single, input["topic"])


async def multi_task(input):
    return await measure(run_multi, input["topic"])


# 평가기: Phoenix가 실행마다 부른다. 필요한 인자(output, expected)만 이름으로 받는다
class Coverage(BaseModel):
    included: list[bool] = Field(description="필수 항목마다 보고서에 그 사실이 있으면 True. 항목 순서대로")


judge = init_chat_model("openai:gpt-6-luna").with_structured_output(Coverage)


async def coverage(output, expected):
    """LLM-as-judge: 필수 항목 중 보고서에 들어 있는 비율"""
    must = expected["must_include"]
    items = "\n".join(f"{i}. {x}" for i, x in enumerate(must, 1))
    verdict = await judge.ainvoke(
        f"보고서에 각 필수 항목의 사실이 들어 있는지 판정하라.\n# 필수 항목\n{items}\n# 보고서\n{output['report']}")
    return sum(verdict.included[:len(must)]) / len(must)


def normalize(url):
    return url.rstrip("/.,")  # 끝의 슬래시와 문장부호 차이를 없앤다


def grounded(output):
    """보고서가 인용한 URL 중 실제 검색 결과에 있었던 비율"""
    cited = {normalize(u) for u in URL.findall(output["report"])}
    if not cited:
        return 0.0
    searched = {normalize(u) for u in output["searched"]}
    return len(cited & searched) / len(cited)


def steps(output):
    return output["steps"]


def searches(output):
    return output["searches"]


def cost_usd(output):
    return output["cost"]


async def upload_dataset(client, n):
    """topics.jsonl의 앞 n개 주제와 필수 항목을 Phoenix 데이터셋으로 올린다."""
    cases = [json.loads(line) for line in Path(__file__).with_name("topics.jsonl").open()][:n]
    return await client.datasets.create_dataset(
        name="research-topics",
        inputs=[{"topic": c["topic"]} for c in cases],
        outputs=[{"must_include": c["must_include"]} for c in cases],
        metadata=[{"id": c["id"]} for c in cases])


CONFIGS = {"single": single_task, "multi": multi_task}
EVALUATORS = [coverage, grounded, steps, searches, cost_usd]


async def run_experiments(client, dataset):
    """같은 데이터셋과 평가기로 두 구성을 동시에 실험한다. 바뀌는 것은 구성뿐이다."""
    experiments = await asyncio.gather(*[
        client.experiments.run_experiment(
            dataset=dataset, task=task, experiment_name=config, evaluators=EVALUATORS,
            concurrency=2,   # 구성마다 2개씩, 두 구성을 동시에 실행
            timeout=600,     # Multi Agent 실행은 몇 분 걸린다(기본 60초)
            retries=0,       # 실패한 실행을 다시 돌리지 않는다(기본 3회)
            print_summary=False)
        for config, task in CONFIGS.items()])
    return dict(zip(CONFIGS, experiments))


def scores_by_run(experiment):
    """평가 결과를 실행별로 묶는다: {실행 ID: {평가기 이름: 점수}}"""
    scores = {}
    for ev in experiment["evaluation_runs"]:
        scores.setdefault(ev.experiment_run_id, {})[ev.name] = ev.result["score"]
    return scores


def collect_rows(dataset, experiments):
    """실행마다 구성, 주제, 점수, 측정값을 한 행으로 모은다."""
    topic_id = {ex["id"]: ex["metadata"]["id"] for ex in dataset.examples}  # 데이터셋 예제 ID → T01 같은 주제 ID
    rows = []
    for config, exp in experiments.items():
        scores = scores_by_run(exp)
        for r in exp["task_runs"]:
            out, s = r["output"], scores[r["id"]]
            rows.append({"config": config, "id": topic_id[r["dataset_example_id"]],
                         "coverage": s["coverage"], "grounded": s["grounded"],
                         "steps": out["steps"], "searches": out["searches"],
                         "cost": out["cost"], "sec": out["sec"],
                         # 성공: 필수 항목을 모두 담고 인용 URL이 모두 검색 결과에 있다
                         "ok": s["coverage"] == 1 and s["grounded"] == 1})
    rows.sort(key=lambda r: (r["id"], r["config"] != "single"))  # 주제별로 single, multi 순
    return rows


def print_table(rows):
    print("\n구성   주제 포함률 인용확인  단계  검색   비용($) 시간(초)  성공")
    for r in rows:
        print(f"{r['config']:<7}{r['id']:<5}{r['coverage']:>6.2f}{r['grounded']:>9.2f}{r['steps']:>6}"
              f"{r['searches']:>6}{r['cost']:>10.4f}{r['sec']:>9.0f}  {'O' if r['ok'] else 'X'}")


def print_summary(rows):
    print()
    for config in CONFIGS:
        rs = [r for r in rows if r["config"] == config]
        ok = sum(r["ok"] for r in rs)
        total = sum(r["cost"] for r in rs)
        per_success = f"${total / ok:.4f}" if ok else "-"
        print(f"{config:<7}성공률 {ok}/{len(rs)} | 평균 단계 {sum(r['steps'] for r in rs) / len(rs):.1f}"
              f" | 평균 비용 ${total / len(rs):.4f} | 성공당 비용 {per_success}")


async def main(n):
    await connect_search()  # 멀티 구성의 researcher들이 함께 쓸 search 서버에 한 번 연결한다
    client = AsyncClient()  # phoenix serve로 띄운 로컬 Phoenix(http://localhost:6006)
    dataset = await upload_dataset(client, n)
    experiments = await run_experiments(client, dataset)
    rows = collect_rows(dataset, experiments)
    print_table(rows)
    print_summary(rows)
    print("\nPhoenix의 Datasets > research-topics > Experiments에서 두 실험을 나란히 비교한다")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
