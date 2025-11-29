from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

load_dotenv()

DATA_DIR = "data"  # put your PDFs here
CHROMA_DIR = "chroma_db"  # Chroma persistence folder

def load_documents():
    docs = []
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_DIR, fname)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if text.strip():
            docs.append((fname, text))
    return docs

def create_vector_store(docs):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="onboarding_docs",
        embedding_function=embed_fn,
    )
    
    ids = []
    texts = []
    metadatas = []
    for i, (fname, content) in enumerate(docs):
        ids.append(f"doc-{i}")
        texts.append(content)
        metadatas.append({"source": fname})
    
    if ids:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        print(f"Ingested {len(ids)} documents into Chroma.")
    else:
        print("No PDF content found to ingest.")

if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    create_vector_store(documents)
    print("Vector store created in 'chroma_db/'.")

