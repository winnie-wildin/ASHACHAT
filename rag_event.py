# rag_event.py
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
import google.generativeai as genai

CSV_PATH = "Events.csv"
EMBEDDING_PATH = "event_embeddings/faiss_index_event"
model = genai.GenerativeModel("gemini-2.0-flash")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

def load_faiss_db_events():
    if os.path.exists(EMBEDDING_PATH):
        return FAISS.load_local(EMBEDDING_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(EMBEDDING_PATH)
        return db

def event_rag(db, query, chat_history=None):
    retrievals = db.similarity_search(query, k=3)
    merged = "\n".join([doc.page_content for doc in retrievals])
    prompt = f"""
You are a helpful assistant for a women's community portal, specializing in providing information about events, such as workshops, conferences, networking sessions, upskilling sessions, or community activities.

Below is context from recent conversation (if any):
{chat_history or ''}

Here are the details about available events:
{merged}

Please answer the user's latest question using ONLY the event information above. 
- If you cannot find any relevant event details in the provided information, simply reply: "I don't know."
- Do not include job listings or unrelated information.

User question: "{query}"

Give a short, clear, and direct answer about events.
"""
    response = model.generate_content(prompt)
    return response.text