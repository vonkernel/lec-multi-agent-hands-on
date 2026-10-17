"""citation checker: 보고서의 주장과 출처 근거를 대조한다"""

import json
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

PROMPT = """너는 인용 검토 담당이다. 보고서에서 출처 번호가 달린 주장마다 출처 근거가 그 주장을 뒷받침하는지 확인한다.
- 근거에 없는 수치, 날짜, 고유명사가 있으면 문제로 적는다.
- 출처 번호가 없는 사실 주장도 문제로 적는다."""


class Review(BaseModel):
    passed: bool = Field(description="모든 주장이 출처 근거로 뒷받침되면 True")
    issues: list[str] = Field(description="문제가 있는 주장과 이유. 없으면 빈 목록")


checker = create_agent("openai:gpt-6-luna", tools=[], system_prompt=PROMPT,
                       response_format=Review, name="citation_checker")


@tool
async def check_citations(runtime: ToolRuntime) -> str:
    """가장 최근 보고서의 주장과 출처를 대조해 검토 결과를 돌려준다."""
    # checker는 보고서와 출처 근거만 본다. 조사 과정과 writer의 의도는 모른다
    report = Path(runtime.state["drafts"][-1]).read_text()
    sources = [s for f in runtime.state["findings"] for s in f["sources"]]
    content = f"# 보고서\n{report}\n\n# 출처 근거\n{json.dumps(sources, ensure_ascii=False, indent=1)}"
    out = await checker.ainvoke({"messages": [{"role": "user", "content": content}]})
    review = out["structured_response"]
    return "통과" if review.passed else "수정 필요\n- " + "\n- ".join(review.issues)
