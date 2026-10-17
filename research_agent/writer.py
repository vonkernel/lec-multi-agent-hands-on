"""writer: 조사 결과로 보고서를 쓰고 artifact 파일로 저장한다"""

import json
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

ARTIFACTS = Path("artifacts")
MAX_REVISIONS = 3  # 5단계: 수정 지시 상한

PROMPT = """너는 보고서 작성 담당이다. 받은 조사 결과만으로 한국어 마크다운 보고서를 쓴다.
- 주장마다 [1], [2]처럼 출처 번호를 달고, 끝의 '## 출처'에 번호, 제목, URL을 적는다.
- 조사 결과에 없는 사실은 쓰지 않는다.
- 이전 보고서와 수정 지시가 있으면 지시한 부분만 고친다. 근거 부족 지적은 문장을 지우기보다 맞는 출처 번호로 바꾼다.
- 보고서 본문만 출력한다."""

writer = create_agent("openai:gpt-6-luna", tools=[], system_prompt=PROMPT, name="writer")


@tool
async def write_report(instructions: str, runtime: ToolRuntime) -> Command | str:
    """조사 결과로 보고서를 쓰거나, 수정 지시에 따라 고쳐 쓴다. 조사 결과는 자동으로 전달된다.
    instructions: 보고서 구성과 강조점, 또는 검토에서 나온 수정 지시"""
    if os.environ.pop("FAIL_WRITER", None):  # 세션 3 3단계: 실습용 오류. 한 번만 난다
        raise TimeoutError("writer 모델 응답 시간 초과 (실습용 오류)")
    drafts = runtime.state.get("drafts", [])
    if len(drafts) > MAX_REVISIONS:  # 5단계: 처음 작성 1번 + 수정 3번까지
        return f"수정 상한({MAX_REVISIONS}회)에 도달했다. 최종 보고서: {drafts[-1]}"
    # 조사 결과는 리드가 옮겨 적지 않는다. researcher가 state에 남긴 것을 그대로 넘긴다
    parts = ["# 조사 결과", json.dumps(runtime.state["findings"], ensure_ascii=False, indent=1),
             "# 지시", instructions]
    if drafts:  # 5단계: 수정이면 이전 보고서를 함께 넘긴다
        parts += ["# 이전 보고서", Path(drafts[-1]).read_text()]
    out = await writer.ainvoke({"messages": [{"role": "user", "content": "\n\n".join(parts)}]})
    report = out["messages"][-1].text

    run_dir = ARTIFACTS / runtime.config["configurable"]["thread_id"]  # 실행 ID별 폴더
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"report_v{len(list(run_dir.glob('report_v*.md'))) + 1}.md"
    path.write_text(report)
    headings = [line[3:] for line in report.splitlines() if line.startswith("## ")]
    return Command(update={
        "drafts": [str(path)],  # 5단계: checker가 읽을 최신 보고서 경로
        "messages": [ToolMessage(f"저장: {path} ({len(report):,}자)\n목차: {' / '.join(headings)}",
                                 tool_call_id=runtime.tool_call_id)],  # 리드에게는 경로와 목차만
    })
