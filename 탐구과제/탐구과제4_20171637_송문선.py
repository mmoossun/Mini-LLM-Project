from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

def extract_text(ai_message):
    return ai_message.content

def make_composed_chain(joke_prompt, analysis_prompt, model, output_parser):
    joke_chain = RunnableLambda(
        lambda inputs: model.invoke(joke_prompt.format_prompt(**inputs)).content
    )
    
    analysis_chain = RunnableLambda(
        lambda inputs: model.invoke(analysis_prompt.format_prompt(**inputs)).content
    )

    composed_chain = RunnableParallel(
        {
            "joke": joke_chain,
            "analysis": RunnableLambda(
                lambda inputs: analysis_chain.invoke({"joke": joke_chain.invoke(inputs)})
            )
        }
    )

    return composed_chain

if __name__ == "__main__":
    load_dotenv()

    joke_prompt = ChatPromptTemplate.from_template("tell me a joke about {topic} in Korean")
    analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")

    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION")
    )
    output_parser = StrOutputParser()

    chain = make_composed_chain(joke_prompt, analysis_prompt, model, output_parser)

    result = chain.invoke({"topic": "beets"})
    print(result)
