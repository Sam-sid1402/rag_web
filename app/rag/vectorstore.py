from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.rag.loader import load_documents
from app.rag.splitter import split_documents

load_dotenv()

PERSIST_DIR = Path("data/chroma_langchain_db")
COLLECTION_NAME = "portfolio_rag"
EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def create_vectorstore(chunks):
    """Create and persist a Chroma vector store from document chunks."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )


def vectorstore_exists() -> bool:
    """Return True when a persisted Chroma database already exists."""
    return (PERSIST_DIR / "chroma.sqlite3").exists()


def build_vectorstore():
    """Build the vector store from local TXT documents."""
    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No TXT documents found in the data/ folder. "
            "Add portfolio documents before building the vector store."
        )

    chunks = split_documents(documents)

    if not chunks:
        raise RuntimeError("Documents were loaded, but no chunks were created.")

    return create_vectorstore(chunks)


def load_vectorstore(create_if_missing: bool = True):
    """
    Load the persisted vector store.

    If create_if_missing=True, the app will automatically build the database from
    data/*.txt when the Chroma database is missing. This makes Render deployment
    safer because generated DB files can be recreated from versioned documents.
    """
    if create_if_missing and not vectorstore_exists():
        return build_vectorstore()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )
