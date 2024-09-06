from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

# 모델 초기화
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # gpt-4o is set by env
    temperature=1.0
)

# 초기 메시지 설정
history = [
    SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
]

def print_messages(messages):
    for message in messages:
        print(f"{message.role}: {message.content}")

def chat():
    print("Chatbot is ready to chat! Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # 사용자 메시지를 히스토리에 추가
        history.append(HumanMessage(content=user_input))
        
        # 모델 호출 및 응답
        model_output = model.invoke(history)
        
        # 모델 응답을 히스토리에 추가
        history.append(AIMessage(content=model_output.content))
        
        # 응답 출력
        print(f"Assistant: {model_output.content}")

if __name__ == "__main__":
    chat()
