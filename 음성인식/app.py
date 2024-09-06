import streamlit as st
import whisper
import speech_recognition as sr
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Whisper 모델 로드
model = whisper.load_model("base")
file_name = "recorded_audio.wav"

# Recognizer 객체 생성
r = sr.Recognizer()

def transcribe_audio(audio_file):
    # Whisper 모델을 사용하여 오디오 파일을 텍스트로 변환
    audio_bytes = audio_file.read()
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    result = model.transcribe(file_name)
    return result['text']

def main():
    st.title("음성 파일 기반 챗봇 및 회의록 작성기")

    # 음성 파일을 텍스트로 변환하는 함수
    def get_audio_input(audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            text = transcribe_audio(audio_file)
        if text:
            return text
        else:
            return None

    # 챗봇 응답을 얻는 함수
    def get_chatbot_response(user_input):
        return f"'{user_input}'"

    def record_audio():
        with sr.Microphone() as source:
            st.write("녹음 시작...")
            audio_data = r.listen(source)
            st.write("녹음 완료!")
            return audio_data

    # 상태 초기화
    if "recorded" not in st.session_state:
        st.session_state.recorded = False
    if "all_transcripts" not in st.session_state:
        st.session_state.all_transcripts = ""
    if "summaries" not in st.session_state:
        st.session_state.summaries = []

    if st.button("음성 녹음 시작") and not st.session_state.recorded:
        audio_data = record_audio()

        # 오디오 파일로 저장
        with open(file_name, "wb") as f:
            f.write(audio_data.get_wav_data())
        
        st.success(f"녹음된 파일이 {file_name}으로 저장되었습니다.")
        st.session_state.recorded = True

    if st.session_state.recorded:
        # 녹음된 파일의 경로를 uploaded_file 변수에 할당
        uploaded_file = file_name

        if uploaded_file is not None:
            user_input = get_audio_input(uploaded_file)
            if user_input is not None:
                st.session_state.all_transcripts += user_input + "\n"
                st.text_area("모든 녹음 내용:", st.session_state.all_transcripts, height=200)
                
                st.session_state.recorded = False

    if st.button("요약하기") and st.session_state.all_transcripts:
        load_dotenv()

        meeting_notes = st.session_state.all_transcripts
        prompt = ChatPromptTemplate.from_template("summarize this meeting minutes '{meeting_notes}' in korean")

        model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("OPENAI_API_VERSION")
        )
        output_parser = StrOutputParser()

        chain = prompt | model | output_parser

        out = chain.invoke({"meeting_notes": meeting_notes})

        # 회의록 요약을 세션 상태에 추가
        st.session_state.summaries.append(out)

        # 모든 요약본을 화면에 표시
        st.header("회의록 요약")
        for idx, summary in enumerate(st.session_state.summaries):
            st.text_area(f"회의록 요약 {idx + 1}:", summary, height=200)

        # 최종 요약본 생성
        final_summary = "\n".join(st.session_state.summaries)

        # 최종 요약본을 화면에 표시
        st.header("최종 요약본")
        st.text_area("최종 요약본:", final_summary, height=200)

        # 텍스트 파일로 저장할 내용
        text_to_save = f"모든 녹음 내용:\n{st.session_state.all_transcripts}\n\n최종 요약본:\n{final_summary}"
        
        # 텍스트 파일 다운로드 버튼
        st.download_button(
            label="텍스트 파일로 저장",
            data=text_to_save,
            file_name="meeting_notes.txt",
            mime="text/plain"
        )

    if st.button("초기화 하기"):
        st.session_state.recorded = False
        st.session_state.all_transcripts = ""
        st.session_state.summaries = []
        st.success("모든 내용이 초기화되었습니다.")

if __name__ == "__main__":
    main()
