from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma
from prompts import (
    chunk_summary_prompt,
    final_summary_prompt
)


load_dotenv()


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

summary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)



# ---------------------------------------------------------
# Load + split PDF
# ---------------------------------------------------------

def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return documents, chunks


# ---------------------------------------------------------
# Create vector database / retriever
# ---------------------------------------------------------

def build_retriever(chunks, k: int = 4):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever


# ---------------------------------------------------------
# Summarize paper
# ---------------------------------------------------------

def summarize_paper(documents) -> str:

    # Larger chunks are used for summarization.
    # We don't need the small 1000-character RAG chunks here.
    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=18000,
        chunk_overlap=500
    )

    summary_chunks = summary_splitter.split_documents(documents)

    partial_summaries = []

    for chunk in summary_chunks:

        prompt = chunk_summary_prompt.invoke({
            "paper_text": chunk.page_content
        })

        response = summary_llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )

        partial_summaries.append(content)

    final_prompt = final_summary_prompt.invoke({
        "partial_summaries": "\n\n".join(partial_summaries)
    })

    final_response = summary_llm.invoke(final_prompt)
    final_content = final_response.content

    if isinstance(final_content, list):
        final_content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in final_content
        )

    return final_content


# ---------------------------------------------------------
# Process uploaded paper
# ---------------------------------------------------------

def process_paper(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4
):

    documents, chunks = load_and_split_pdf(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    retriever = build_retriever(
        chunks=chunks,
        k=k
    )

    paper_summary = summarize_paper(documents)

    return retriever, paper_summary


# ---------------------------------------------------------
# Old function kept for notebook/testing purposes
# ---------------------------------------------------------

def create_retriever(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4
):

    _, chunks = load_and_split_pdf(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return build_retriever(
        chunks=chunks,
        k=k
    )


# ---------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------

def retrieve_context(task: str, retriever) -> str:

    docs = retriever.invoke(task)

    context = "\n\n".join(
        f"""Source {i + 1}
Page: {doc.metadata.get("page_label", "Unknown")}

{doc.page_content}"""
        for i, doc in enumerate(docs)
    )

    return context