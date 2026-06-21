import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.rag.prompt import build_prompt

load_dotenv()

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_llm(temperature: float = 0.0):
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
    )


def format_chat_history(chat_history):
    if not chat_history:
        return ""
    return "\n".join(chat_history)


def clean_source_name(source: str) -> str:
    return Path(source).name if source else "unknown"


def format_context(chunks_with_scores):
    context_parts = []
    sources = []

    for chunk, score in chunks_with_scores:
        source = clean_source_name(chunk.metadata.get("source", "unknown"))
        sources.append(source)

        context_parts.append(
            f"""
Source: {source}
Retrieval distance: {score}

Content:
{chunk.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts), sorted(set(sources))


def rewrite_question(question, history_text):
    if not history_text:
        return question

    rewrite_prompt = f"""
Rewrite the current user question as a standalone question.

Use the previous conversation only to resolve references like "it", "this", "that", "the first one", "the second one", "the project", or "what else".

Do not answer the question.
Return only the rewritten standalone question.

Previous conversation:
{history_text}

Current user question:
{question}

Standalone question:
"""

    response = get_llm(temperature=0.0).invoke(rewrite_prompt)
    return response.content.strip()


def answer_question(vector_store, question, chat_history=None):
    chat_history = chat_history or []
    history_text = format_chat_history(chat_history)

    standalone_question = rewrite_question(
        question=question,
        history_text=history_text,
    )

    retrieved_chunks = vector_store.similarity_search_with_score(
        standalone_question,
        k=TOP_K,
    )

    context, sources = format_context(retrieved_chunks)

    prompt = build_prompt(
        context=context,
        question=question,
        chat_history=history_text,
    )

    response = get_llm(temperature=0.2).invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
        "standalone_question": standalone_question,
        "retrieved_chunks": len(retrieved_chunks),
    }
