"""search MCP 서버: Tavily 검색 함수를 search, fetch 도구로 제공한다"""

from mcp.server.mcpserver import MCPServer

from tavily_search import fetch, search

mcp = MCPServer("search")
mcp.tool()(search)  # 함수 이름, docstring, 타입 힌트가 MCP 도구 스키마가 된다
mcp.tool()(fetch)

if __name__ == "__main__":
    mcp.run(transport="stdio")  # stdin/stdout으로 요청을 주고받는다
