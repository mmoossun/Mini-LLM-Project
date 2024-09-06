from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os


def ask_something(chain, query):
    print(f"User: {query}")
    chain_output = chain.invoke({"input": query})
    print(f"LLM: {chain_output}")
    return

def init_retriever(filepath):
    with open(filepath, encoding='utf-8') as f:
        frog_txt = f.read()

    frog_document = Document(
        page_content=frog_txt,
        metadata={"source": "개구리 왕자 - 방정환 역"}
    )

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", ","],
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )

    split_docs = recursive_text_splitter.split_documents([frog_document])

    embedding_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    chroma = Chroma("vector_store")
    vector_store = chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model
    )

    retriever = vector_store.as_retriever(search_type="similarity")
    return retriever
def init_agent(retriever):
    azure_model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION")
    )
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        azure_model, retriever, contextualize_q_prompt
    )

    qa_system_prompt_str = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you cannot find the answer in the retrieved context, try to find it in chat history.
    If you don't know the answer after all, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Answer for the question in Korean.
    
    {context} """.strip()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt_str),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            MessagesPlaceholder(variable_name="context"),
        ]
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="txt_search",
        # 도구에 대한 설명을 자세히 기입해야 합니다!!!
        description="개구리 왕자에 대한 정보를 검색합니다.",
    )
    tools = [retriever_tool]

    agent = create_openai_functions_agent(azure_model, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
    
    memory = ConversationBufferMemory(
        chat_memory=InMemoryChatMessageHistory(),        
        return_messages=True
    )
    load_context_runnable = RunnablePassthrough().assign(
        chat_history=RunnableLambda(lambda x:memory.chat_memory.messages)
    )
    def save_context(agent_output):
        memory.chat_memory.add_user_message(agent_output["input"])
        memory.chat_memory.add_ai_message(agent_output["output"])
        return agent_output["output"]

    def extract_context(input):
        query = input["input"]
        retrieved_docs = retriever.get_relevant_documents(query)
        context = [SystemMessage(content=doc.page_content) for doc in retrieved_docs]
        input["context"] = context
        return input
    extract_context_runnable = RunnableLambda(extract_context)
    save_context_runnable = RunnableLambda(save_context)
    agent_chain = load_context_runnable | extract_context_runnable | agent_executor | save_context_runnable

    return agent_chain
if __name__ == "__main__":
    load_dotenv()

    filepath = "/root/LLM_Bootcamp/LangChain_Class/Retrieval/frog_prince.txt"

    retriever = init_retriever(filepath)
    agent_chain = init_agent(retriever)
    
    
    print("Chatbot is ready to chat! Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        ask_something(agent_chain, user_input)
