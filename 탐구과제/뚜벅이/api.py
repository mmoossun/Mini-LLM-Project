import os
import re
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

# Google Maps API 설정
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

def extract_keywords(text):
    """
    Extracts the keywords from the ChatGPT generated text.
    :param text: The answer from ChatGPT, like 'Keywords: Golden Heart, Beating, 50th Anniversary, Promo, Promotion video.'
    """
    text = text.lower()
    expression = r".*keywords:(.+?)$"
    if re.search(expression, text):
        keywords = re.sub(expression, r"\1", text, flags=re.S)
        if keywords is not None and len(keywords) > 0:
            return [re.sub(r"\.$", "", k.strip()) for k in keywords.strip().split(',')]
    return []

def extract_keywords_from_chat(chat, input_text):
    """
    Sends a chat question to ChatGPT and returns its output.
    :param chat: The object which communicates under the hood with ChatGPT.
    :param input_text: The input text from the user
    """
    resp = chat.invoke([
                SystemMessage(content=
                        "You extract the main keywords in the text and extract these into a comma separated list. Please prefix the keywords with 'Keywords:'"),
                HumanMessage(content=input_text)
            ])
    answer = resp.content
    return answer

def search_place(query):
    """
    구글 지도 API를 사용하여 특정 장소를 검색합니다.
    :param query: 검색할 장소의 이름 또는 키워드
    :return: 검색 결과
    """
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": google_maps_api_key,
        "language": "ko"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    """
    Main function to process user input, extract keywords, and search places
    """
    chat = AzureChatOpenAI(
        azure_deployment=model_name,
        openai_api_key=api_key,
        api_version=api_version,
        temperature=0
    )

    while True:
        user_input = input("Enter a sentence to extract keywords and search places (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        try:
            # 키워드 추출
            answer = extract_keywords_from_chat(chat, user_input)
            extracted_keywords = extract_keywords(answer)
            print(f"Extracted Keywords: {extracted_keywords}")
            
            # 추출된 키워드를 통합하여 장소 검색
            if extracted_keywords:
                combined_query = ' '.join(extracted_keywords)
                result = search_place(combined_query)
                if result and "results" in result:
                    places = result["results"]
                    print(f"Results for combined keywords: {combined_query}")
                    for place in places:
                        name = place.get('name', 'N/A')
                        address = place.get('formatted_address', 'N/A')
                        rating = place.get('rating', 'N/A')
                        open_now = place.get('opening_hours', {}).get('open_now', 'N/A')
                        print(f"Name: {name}, Address: {address}, Rating: {rating}, Open Now: {open_now}")
                else:
                    print("No results found for combined keywords.")
            else:
                print("No keywords extracted.")
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
