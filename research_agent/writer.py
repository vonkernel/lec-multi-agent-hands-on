"""writer: 조사 결과로 보고서를 쓰고 artifact 파일로 저장한다"""

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

ARTIFACTS = Path("artifacts")

PROMPT = """너는 보고서 작성 담당이다. 받은 조사 결과만으로 한국어 마크다운 보고서를 쓴다.
- 주장마다 [1], [2]처럼 출처 번호를 달고, 끝의 '## 출처'에 번호, 제목, URL을 적는다.
- 조사 결과에 없는 사실은 쓰지 않는다.
- 보고서 본문만 출력한다."""

writer = create_agent("openai:gpt-6-luna", tools=[], system_prompt=PROMPT, name="writer")


@tool
async def write_report(instructions: str, runtime: ToolRuntime) -> str:
    """조사 결과로 보고서를 쓰게 한다. 조사 결과는 자동으로 전달된다.
    instructions: 보고서 구성과 강조점"""
    # 조사 결과는 리드가 옮겨 적지 않는다. researcher가 state에 남긴 것을 그대로 넘긴다
    parts = ["# 조사 결과", json.dumps(runtime.state["findings"], ensure_ascii=False, indent=1),
             "# 지시", instructions]
    out = await writer.ainvoke({"messages": [{"role": "user", "content": "\n\n".join(parts)}]})
    report = out["messages"][-1].text

    run_dir = ARTIFACTS / runtime.config["configurable"]["thread_id"]  # 실행 ID별 폴더
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"report_v{len(list(run_dir.glob('report_v*.md'))) + 1}.md"
    path.write_text(report)
    headings = [line[3:] for line in report.splitlines() if line.startswith("## ")]
    return f"저장: {path} ({len(report):,}자)\n목차: {' / '.join(headings)}"  # 리드에게는 경로와 목차만
