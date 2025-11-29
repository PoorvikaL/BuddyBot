import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

CHROMA_DIR = "chroma_db"

SYSTEM_PROMPT = """
You are an Employee Onboarding Copilot.

You MUST:
- Answer using ONLY the provided context from company onboarding, tools, and HR documents.
- If something is not present or is unclear in the context, say you are not sure and suggest asking HR or the hiring manager.
- Be friendly, clear, and practical.
- Focus on helping new hires understand forms, tools, trainings, and first weeks at the company.
"""


def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectordb


def build_rag_chain():
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def rag(question: str) -> str:
        # retriever is a Runnable in new LangChain, use invoke()
        docs = retriever.invoke(question)
        context = format_docs(docs)
        messages = prompt.format_messages(context=context, input=question)
        response = llm.invoke(messages)
        return response.content

    return llm, vectordb, rag
