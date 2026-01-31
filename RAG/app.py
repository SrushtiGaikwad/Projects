import os
import certifi

# ===============================
# ENV & CONFIG
# ===============================

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["GEMINI_API_KEY"] = "AIzaSyCCMvJdc343e21sSObV0oWKdO727T-If8M"

import google.generativeai as genai
genai.configure(api_key=os.environ["AIzaSyCCMvJdc343e21sSObV0oWKdO727T-If8M"])

# ===============================
# IMPORTS
# ===============================

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ===============================
# STEP 1: FETCH TRANSCRIPT
# ===============================

def fetch_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()

        fetched_transcript = api.fetch(
            video_id= "xOCaT4KYvqs",
            languages=["en"],
            preserve_formatting=False
        )

        transcript = " ".join(
            snippet.text for snippet in fetched_transcript.snippets
        )

        return transcript

    except TranscriptsDisabled:
        return ""

# ===============================
# STEP 2: BUILD VECTOR STORE
# ===============================

def build_vector_store(transcript: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ["AIzaSyCCMvJdc343e21sSObV0oWKdO727T-If8M"],
        batch_size=5
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# ===============================
# STEP 3: SETUP LLM & PROMPT
# ===============================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.environ["AIzaSyCCMvJdc343e21sSObV0oWKdO727T-If8M"]
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

gen_model = genai.GenerativeModel("gemini-2.5-flash")

# ===============================
# STEP 4: RAG PIPELINE FUNCTION
# ===============================

def rag_qa(question: str, retriever) -> str:
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "I don't know. The answer is not present in the video."

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    response = gen_model.generate_content(final_prompt.text)
    return response.text.strip()

# ===============================
# INITIALIZATION (ONCE)
# ===============================

def initialize_rag(video_id: str):
    transcript = fetch_transcript(video_id)

    if not transcript:
        raise ValueError("Transcript not available for this video")

    vector_store = build_vector_store(transcript)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever
