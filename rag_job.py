# rag_job.py
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
import google.generativeai as genai

CSV_PATH = "DataAnalyst.csv"
EMBEDDING_PATH = "embeddings/faiss_index"
model = genai.GenerativeModel("gemini-2.0-flash")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

def load_faiss_db():
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

def rag(db, query,chat_history=None):
    retrievals = db.similarity_search(query, k=3)
    merged = "\n".join([doc.page_content for doc in retrievals])
    prompt = f"""
You are a helpful job search assistant for a job search website.

Here is the recent chat history between you and the user:
{chat_history or ''}

You have access to the following job listing information:
{merged}

Based on this data AND the user's chat history, answer the current question as best as you can—by providing relevant job information, filtering for location, role, or skills based on the user's request. If you cannot find any relevant jobs in the provided data, respond with "I don't know".

Current user query: "{query}"

If possible, summarize or list suitable job options pulled from the data above.
"""
    response = model.generate_content(prompt)
    return response.text