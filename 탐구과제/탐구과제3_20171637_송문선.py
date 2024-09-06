from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

# 모델 초기화
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # gpt-4o is set by env
    temperature=1.0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


output_parser = StrOutputParser()
chain = prompt | model | output_parser

memory = ConversationBufferMemory(
            chat_memory=InMemoryChatMessageHistory(),
            return_messages=True #대화 기록이 메시지 객체(HumanMessage, AIMessage등)의 리스트로 반환
        )


def chat():
    print("Chatbot is ready to chat! Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # 대화 기록 가져오기
        chat_history = memory.chat_memory.messages

        # 체인 실행 
        output = chain.invoke({
            "input": user_input,
            "chat_history": chat_history        
        })
        
        # 모델 응답 출력
        print(f"AI: {output}")

        # 메모리에 사용자 입력과 AI 응답 추가
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(output)

    # 대화 종료 시 대화 히스토리 출력
    print("\nConversation History:")
    print(memory.chat_memory.messages)

if __name__ == "__main__":
    chat()
