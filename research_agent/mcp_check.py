"""세션 2 [추가]: search MCP 서버에 연결해 도구 목록을 받고, search를 한 번 호출한다"""

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain.mcp import MCPAdapter

SEARCH_SERVER = {"mcpServers": {"search": {
    "command": sys.executable,
    "args": [str(Path(__file__).with_name("search_server.py"))],
    "env": {"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},  # stdio 서버는 환경 변수를 물려받지 않는다
}}}


async def main():
    async with MCPAdapter(SEARCH_SERVER) as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}  # 서버가 제공하는 도구를 LangChain 도구로 받는다
        print("도구:", list(tools))
        blocks = await tools["search"].ainvoke({"query": "주 52시간 근무제 시행 시기"})
        found = blocks[0]["text"]  # MCP 도구 결과는 콘텐츠 블록 목록. search는 텍스트 블록 하나를 돌려준다
        titles = [r["title"] for r in json.loads(found)["results"][:2]]
        print(f"search: {len(found):,}자 | {titles}")


if __name__ == "__main__":
    asyncio.run(main())
