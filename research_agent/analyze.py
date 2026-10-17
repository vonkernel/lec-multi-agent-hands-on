"""세션 4 2단계: 실행 기록에서 trajectory를 펼쳐 낭비된 호출을 찾는다"""

import json
import sys
from collections import Counter
from itertools import groupby

from agentevals.trajectory.match import create_trajectory_match_evaluator
from langchain.messages import AIMessage, HumanMessage

REQUIRED = ["research", "write_report", "check_citations"]  # 리드가 거쳐야 할 도구
# superset: 기준 도구를 모두 호출했는지만 본다. 순서, 추가 호출, 인자는 보지 않는다
covers = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")


def as_trajectory(tool_names):
    """도구 이름 목록을 agentevals가 읽는 메시지 형식으로 바꾼다."""
    calls = [{"name": n, "args": {}, "id": str(i)} for i, n in enumerate(tool_names)]
    return [HumanMessage("주제"), AIMessage("", tool_calls=calls)]


# 리드의 위임
def show_lead_steps(lead_calls):
    """리드가 부른 도구를 순서대로 보여 준다. 연달아 부른 같은 도구는 횟수로 묶는다."""
    steps = []
    # 예: [research, research, write_report] → research x2, write_report x1
    for name, group in groupby(lead_calls):
        steps.append(f"{name} x{len(list(group))}")
    print("리드 단계:", " → ".join(steps))


def has_required_steps(lead_calls):
    """리드가 필수 도구를 모두 한 번 이상 불렀으면 True"""
    result = covers(outputs=as_trajectory(lead_calls), reference_outputs=as_trajectory(REQUIRED))
    return result["score"]


# researcher의 검색
def show_searches(searches):
    """researcher마다 search와 fetch를 몇 번 불렀는지 보여 준다."""
    for s in searches:
        kinds = Counter(c["name"] for c in s["calls"])
        print(f"  researcher | {s['topic']} | search {kinds['search']}, fetch {kinds['fetch']}")


def find_duplicates(searches):
    """같은 도구를 같은 인자로 두 번 이상 부른 호출과 그 횟수"""
    counts = Counter()
    for s in searches:
        for call in s["calls"]:
            # dict는 Counter의 키가 될 수 없어 키를 정렬한 JSON 문자열로 바꿔 센다
            counts[json.dumps(call, ensure_ascii=False, sort_keys=True)] += 1
    return {call: n for call, n in counts.items() if n > 1}


def analyze(record):
    """리드 단계, 필수 단계 여부, researcher별 검색, 중복 호출을 출력한다."""
    show_lead_steps(record["lead_calls"])
    print("필수 단계를 모두 거쳤나:", has_required_steps(record["lead_calls"]))
    show_searches(record["searches"])
    for call, n in find_duplicates(record["searches"]).items():
        print(f"  중복 호출 x{n}: {call}")
    total = sum(len(s["calls"]) for s in record["searches"])
    print(f"검색 호출 {total}번, 보고서 {len(record['drafts'])}개 버전")


if __name__ == "__main__":
    analyze(json.load(open(sys.argv[1])))
