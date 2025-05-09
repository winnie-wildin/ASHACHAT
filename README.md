# Asha-bot
 is an intelligent, ethical, and memory-enhanced career assistant built with Streamlit, FAISS, HuggingFace embeddings, and Google Gemini. It goes beyond simple Q&amp;A by combining local job/event data with real-time job search from the TheirStack API — while actively guarding against gender bias to ensure an inclusive user experience.

✨ Features
📚 Memory-Enhanced Chat (Summarization):
Recaps the conversation into a smart memory after every few turns to stay focused on your evolving needs.

📈 RAG (Retrieval-Augmented Generation) Engine:
Mixes local FAISS search results with real-time job listings using Gemini prompting for hyper-relevant answers.

🌍 Live Job Fetching (Rapid API):
Instantly pulls the latest open roles based on your query.

🛡️ Ethical Guardrails:
Detects and reframes gender-biased or discriminatory queries with supportive, positive responses.

⚡ Lightning-fast Gemini Flash Model (2.0):
Ultra-efficient prompting and content generation thanks to Google's Gemini 2.0 Flash.

🛠 Easily Configurable:
API keys, models, and settings are loaded dynamically with .env for fast setup.

🏗️ Tech Stack
>Streamlit — Interactive app frontend

>Google Gemini Flash API — LLM-powered understanding & response

>FAISS — Facebook AI Similarity Search

>HuggingFace MiniLM Embeddings — Compact yet powerful semantic search

>RAPID API — Real-time job database

>LangChain Unstructured — Document ingestion

>Python dotenv — Safe environment config management
