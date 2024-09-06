import streamlit as st
from PIL import Image
import requests
import base64
import os
import sqlite3
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key)

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

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_text_from_image(image_data):
    response = client.completions.create(
        model="gpt-4",
        prompt=system_prompt + "\n" + f"data:image/jpeg;base64,{image_data}",
        max_tokens=500,
        temperature=0.0
    )
    return response.choices[0].text.strip()

def alter_table_add_column(conn, column_name, column_type):
    c = conn.cursor()
    try:
        c.execute(f"ALTER TABLE extracted_data ADD COLUMN {column_name} {column_type}")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            pass  # Column already exists, ignore error
        else:
            raise e

def save_to_sqlite(extracted_text):
    conn = sqlite3.connect('extracted_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS extracted_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT)''')
    # 테이블에 열 추가
    alter_table_add_column(conn, "extracted_text", "TEXT")
    # 디버깅: 데이터 확인
    st.write("Saving data to SQLite:", extracted_text)
    c.execute('''INSERT INTO extracted_data (extracted_text)
                 VALUES (?)''',
              (extracted_text,))
    conn.commit()
    conn.close()
    st.success("Data saved to SQLite database successfully!")

def fetch_from_sqlite():
    conn = sqlite3.connect('extracted_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM extracted_data')
    rows = c.fetchall()
    conn.close()
    return rows

def main():
    st.title("Image Text Extractor")
    st.write("Upload an image or provide an image URL to extract text.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter an image URL:")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image_data = encode_image(image)
        with st.spinner('Extracting text...'):
            extracted_text = extract_text_from_image(image_data)
        st.write("Extracted Text:")
        st.text(extracted_text)

        save_to_sqlite(extracted_text)

    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            image_data = encode_image(image)
            with st.spinner('Extracting text...'):
                extracted_text = extract_text_from_image(image_data)
            st.write("Extracted Text:")
            st.text(extracted_text)

            save_to_sqlite(extracted_text)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    st.write("### Extracted Data from Database")
    rows = fetch_from_sqlite()
    if rows:
        for row in rows:
            st.write(f"ID: {row[0]}", end=', ')
            st.write(f"Extracted Text: {row[1]}")
            st.write()  # 줄바꿈
    else:
        st.write("No data found in the database.")

if __name__ == "__main__":
    main()
