"""세션 1 1단계: 실습 패키지와 API 키 확인"""

import os
import sys
from importlib.metadata import version

print("Python", sys.version.split()[0])
for package in ("langchain", "langgraph", "mcp", "tavily-python", "arize-phoenix"):
    print(package, version(package))
print({key: key in os.environ for key in ("OPENAI_API_KEY", "TAVILY_API_KEY")})
