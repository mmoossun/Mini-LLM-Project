
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64


api_key = os.getenv("OPENAI_API_KEY")

# Set API key
client = OpenAI()



system_prompt = """
사진에 있는 글자를 모두 출력해. 너의 의견을 적지마. 단순한 글자들의 나열만 출력해.너가 지어내면 안돼.

예를 들면, 직업, 전화번호, 이메일의 나열만 보여줘.무조건 JSON형태로 출력해야해.
직책은 직책, 이름은 이름에 맞게 들어가야 한다는 것을 명심해.
직업: 엔지니어
직책: 대표
전화번호: 000-0000-0000
이메일: aaaa@aaaaa.com
이름: 홍길동

"""
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("card.jpg")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "사진에 있는 글자를 모두 출력해"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }  
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "사진에 있는 글자를 모두 출력해"},
            {"type": "image_url", "image_url": {
                "url": "https://www.developerfastlane.com/img/blog/streamlit/cat.webp"}
            }
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)