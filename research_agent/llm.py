"""모든 Agent가 같이 쓰는 모델. 키는 다른 과정과 겹치지 않는 이름의 환경 변수에서 읽는다"""

import os

from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-6.1-terra", api_key=os.environ["MULTI_AGENT_OPENAI_API_KEY"])
