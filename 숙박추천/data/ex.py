import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# CSV 파일 읽기
csv_path = "/root/LLM_Bootcamp/LangChain_Class/Retrieval/combined_dataset.csv"
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 전체 컬럼의 텍스트 데이터 결합
def combine_columns(row):
    return ' '.join(row.astype(str).values)

df['combined_text'] = df.apply(combine_columns, axis=1)
text_data = df['combined_text'].tolist()

# 불용어 리스트
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다', 'nan']

# 불용어 제거 함수
def remove_stopwords(text):
    pattern = re.compile(r'\b(' + '|'.join(stopwords) + r')\b\s*')
    return pattern.sub('', text)

# 불용어 제거
filtered_text_data = [remove_stopwords(text) for text in text_data]

# Document 객체로 변환
documents = [Document(page_content=text, metadata={"source": "숙박업소 데이터셋"}) for text in filtered_text_data]

# 100개의 데이터 샘플링
sampled_documents = documents[:100]

# 텍스트 분할
chunk_size = 500  # chunk_size를 키움
chunk_overlap = 50  # chunk_overlap 값을 설정
recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", ".", ","],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)

start_time = time.time()
split_documents = recursive_text_splitter.split_documents(sampled_documents)
print(f"Time taken to split documents: {time.time() - start_time} seconds")

# Embedding
embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-large"
)

def get_embeddings_with_retry(documents, model, max_retries=5, delay=60):
    for attempt in range(max_retries):
        try:
            return model.embed_documents(documents)
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                print(f"Rate limit error occurred. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise

def batch_process(documents, batch_size=100, max_workers=5):
    batched_embeddings = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_content = [doc.page_content for doc in batch]
            futures.append(executor.submit(get_embeddings_with_retry, batch_content, embedding_model))
        
        for future in as_completed(futures):
            batched_embeddings.extend(future.result())
    
    return batched_embeddings

start_time = time.time()
embeddings = batch_process(split_documents, batch_size=100, max_workers=5)
print(f"Time taken to generate embeddings: {time.time() - start_time} seconds")

# 결과 저장 (예: CSV 파일로 저장)
embedding_df = pd.DataFrame(embeddings)
embedding_df.to_csv("/root/LLM_Bootcamp/LangChain_Class/Retrieval/embedded_dataset_sample.csv", index=False)

print(f"Number of Embed list of texts : {len(embeddings)}")
print(f"Sample Vector : {embeddings[0][:5]}")
print(f"Length of Sample Vector {len(embeddings[0])}")
