# AI Agent 설계 심화 for Multi Agent: 실습

과정 「AI Agent 설계 심화 for Multi Agent」의 실습 저장소다. 실습은 GitHub Codespaces에서 브라우저로 진행한다.

## 실습 개요

일반 주제를 받아 웹을 검색하고, 출처가 달린 한국어 마크다운 보고서를 쓰는 리서치 Agent를 만든다. 단일 Agent로 시작해 리드와 하위 Agent로 나누고, 사람 승인과 추적·평가를 붙인다.

| 항목 | 내용 |
|---|---|
| 실행 형태 | 터미널 CLI |
| 프레임워크 | LangChain 1.x, LangGraph 1.x |
| 구성 | 리드(감독) + researcher + writer + citation checker. 하위 Agent를 도구로 감싸 부른다 |
| 모델 | 모든 Agent: `openai:gpt-6-luna` |
| 검색 | Tavily. 직접 만든 MCP 서버로 연결한다 |
| 결과물 | `artifacts/<실행 ID>/report_v<n>.md` |
| 개발 환경 | GitHub Codespaces (브라우저) |

## 세션별 진행

| 세션 | 만드는 것 |
|---|---|
| 1. 에이전트 고급 설계 | 검색 도구를 직접 쓰는 단일 ReAct Agent로 보고서를 만든다. 검색 결과가 쌓이며 입력 토큰이 늘어나는 것을 확인한다 |
| 2. 멀티 에이전트 시스템 개발 | 리드가 researcher를 병렬로 부르고, writer가 보고서를 쓰고, citation checker가 출처를 검토하는 Multi Agent로 나눈다. 검색 도구를 MCP 서버로 분리한다 |
| 3. 고급 제어 기법과 컨텍스트 관리 | 조사 계획을 사람이 승인하고, 계획 단계로 돌아가 다시 실행하고, 오류로 멈춘 실행을 이어서 실행한다. 리드가 받는 토큰을 줄인다 |
| 4. 에이전트 운영 및 평가 | Phoenix로 실행을 추적하고, 낭비된 호출을 찾고, 같은 평가 세트로 단일 Agent와 Multi Agent를 비교한다 |

## 진행 방식

- 이 저장소에는 개발 환경 설정(`.devcontainer/`)만 있고 실습 코드는 없다.
- 코드는 강의에서 안내하는 순서대로 Codespace에서 직접 입력하며 작성한다.
- 각 세션은 실행 가능한 상태로 끝난다.

## 시작하기

### 1. Tavily API 키 발급

1. [Tavily](https://app.tavily.com)에 이메일로 가입한다. 카드 등록은 필요 없다.
2. 대시보드에서 API 키를 복사해 둔다. 무료로 매월 1,000크레딧이 주어지고, 실습 검색 한 번은 1크레딧이다.

### 2. Codespace 만들기

1. 이 저장소 페이지에서 **Code**를 누르고 **Codespaces** 탭을 연다.
2. 탭 오른쪽 위의 `…` 메뉴에서 **New with options**를 고른다.
3. **Branch**가 `main`인지 확인한다.
4. 아래쪽 입력란에 두 키를 넣는다. 입력한 값은 본인 계정의 Codespaces secret으로 저장된다.
   - `OPENAI_API_KEY`: 강의에서 안내하는 API 키
   - `TAVILY_API_KEY`: 1단계에서 발급한 본인 키
5. **Create codespace**를 누른다.

- 처음 만들 때 패키지 설치가 자동으로 진행된다. 터미널에서 설치가 끝날 때까지 기다린다.

### 3. 환경 확인

Codespace 터미널에서 실행한다.

```bash
python --version
python -c "import os; print({k: k in os.environ for k in ('OPENAI_API_KEY', 'TAVILY_API_KEY')})"
python -c "import langchain, langgraph; print(langchain.__version__)"

# 예상 출력
# Python 3.14.<패치>
# {'OPENAI_API_KEY': True, 'TAVILY_API_KEY': True}
# 1.4.4
```

### 4. 다시 열기

- 다음 세션에는 새로 만들지 않고 같은 Codespace를 다시 연다. 작성한 코드가 그대로 남아 있다.
- 만든 Codespace 목록은 저장소의 **Code** → **Codespaces** 탭에서 볼 수 있다.

## 개발 환경

| 항목 | 버전 |
|---|---|
| Python | 3.14 |
| `langchain[mcp]` | 1.4.4 |
| `langchain-core` | 1.6.9 |
| `langchain-openai` | 1.7.0 |
| `langgraph` | 1.2.14 |
| `fastmcp` | 4.1.0 |
| `mcp` | 2.3.0 |
| `tavily-python` | 0.8.5 |
| `agentevals` | 0.0.9 |
| `arize-phoenix` | 20.21.3 |
| `openinference-instrumentation-langchain` | 0.1.79 |

- 설정은 [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)에 있다.
- 세션 4의 Phoenix 화면은 Codespace의 **Ports** 탭에서 6006 포트로 연다.

## 자주 발생하는 문제

| 증상 | 원인 | 조치 |
|---|---|---|
| 환경 확인에서 `False`가 나온다 | API 키를 넣지 않았거나, Codespace를 연 뒤에 secret을 등록했다 | [Codespaces 설정](https://github.com/settings/codespaces)에 두 키가 있고 이 저장소에 접근이 허용되어 있는지 확인한다. 그다음 Codespace를 멈췄다가 다시 연다 |
| `langchain` import 오류 | 패키지 설치가 끝나기 전에 실행했다 | 터미널에서 설치가 끝난 뒤 다시 실행한다 |
| `LLMChain` 등 import 오류 | 인터넷의 LangChain 0.x 예제를 썼다 | 강의 코드(LangChain 1.x)를 따른다 |
