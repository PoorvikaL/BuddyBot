import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CHROMA_DIR = "chroma_db"

SYSTEM_PROMPT = """
You are an Employee Onboarding Copilot.

You MUST:

Answer using ONLY the provided context from company onboarding, tools, and HR documents.

If something is not present or is unclear in the context, say you are not sure and suggest asking HR or the hiring manager.

Be friendly, clear, and practical.

Focus on helping new hires understand forms, tools, trainings, and first weeks at the company.
"""

def build_rag_chain():
    """
    Returns a simple function rag(question: str) -> str
    that:
    - queries ChromaDB for relevant chunks
    - calls Gemini to answer using those chunks as context
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("onboarding_docs")
    
    # Updated model name
    model = genai.GenerativeModel("gemini-2.0-flash")  # same as planner.py

    def rag(question: str) -> str:
        # Retrieve top-k docs from Chroma
        try:
            results = collection.query(query_texts=[question], n_results=4)
            docs = results.get("documents", [[]])
            context = "\n\n".join(docs[0]) if docs and docs[0] else "No relevant context found."
        except Exception:
            context = "No relevant context found."
        
        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer using only the context above. If you are not sure, say so and suggest asking HR."""
        
        resp = model.generate_content(prompt)
        return resp.text or "Sorry, I could not generate an answer."
    
    return rag
