"""세션 1 실습: 단일 ReAct 리서치 Agent"""

import sys
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool

from tavily_search import fetch, search

PROMPT = """너는 리서치 보고서를 쓰는 Agent다.
- 주제를 하위 질문 3-4개로 나누고, 질문마다 search로 검색한다. 검색 결과의 본문 일부로 부족할 때만 fetch로 본문을 읽는다.
- 보고서는 한국어 마크다운으로 쓴다. 주장마다 [1], [2]처럼 출처 번호를 달고, 끝의 '## 출처'에 번호, 제목, URL을 적는다.
- 검색 결과에 없는 내용은 쓰지 않는다."""

# 함수 이름, docstring, 타입 힌트가 모델이 보는 도구 설명과 인자 스키마가 된다
agent = create_agent("openai:gpt-6-luna", tools=[tool(search), tool(fetch)],
                     system_prompt=PROMPT, name="single_agent")


def save(report):
    """보고서를 artifacts/<실행 ID>/report.md에 저장한다."""
    path = Path("artifacts") / datetime.now().strftime("%m%d-%H%M%S") / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print("저장:", path)


def main(topic):
    result = agent.invoke({"messages": [{"role": "user", "content": topic}]})
    report = result["messages"][-1].text  # 마지막 AI 메시지가 보고서
    print(report)
    save(report)


if __name__ == "__main__":
    main(sys.argv[1])
