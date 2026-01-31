import streamlit as st
from app import initialize_rag, rag_qa

st.title("YouTube RAG QA Bot 🎥🤖")

video_id = st.text_input("Enter YouTube Video ID")
question = st.text_input("Ask a question")

if "retriever" not in st.session_state and video_id:
    with st.spinner("Processing video..."):
        st.session_state.retriever = initialize_rag(video_id)

if question and "retriever" in st.session_state:
    answer = rag_qa(question, st.session_state.retriever)
    st.write("### Answer")
    st.write(answer)
