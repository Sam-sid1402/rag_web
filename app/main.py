from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag.chain import answer_question
from app.rag.memory import add_message, clear_history, get_history
from app.rag.vectorstore import load_vectorstore

app = FastAPI(
    title="Semyon Portfolio RAG API",
    version="0.1.0",
    description="A portfolio RAG API that answers questions about Semyon Sidorov's ML projects, skills, and experience.",
)

vector_store = load_vectorstore(create_if_missing=True)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Semyon Portfolio RAG API is running",
    }


@app.post("/ask")
def ask(request: AskRequest):
    history = get_history(request.session_id)

    result = answer_question(
        vector_store=vector_store,
        question=request.question,
        chat_history=history,
    )

    add_message(request.session_id, "User", request.question)
    add_message(request.session_id, "Assistant", result["answer"])

    return result


@app.post("/clear-memory")
def clear_memory(request: ClearMemoryRequest):
    clear_history(request.session_id)

    return {
        "status": "ok",
        "message": f"Memory cleared for session_id={request.session_id}",
    }
