from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"


def load_documents():
    docs = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, fname)
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = fname
            docs.extend(pdf_docs)
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb




if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document pages.")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    create_vector_store(chunks)
    print("Vector store created in 'chroma_db/'.")
