from app.rag.vectorstore import load_vectorstore
from app.rag.chain import answer_question


def main():
    vector_store = load_vectorstore()

    print("Semyon's Portfolio RAG Assistant")
    print("Ask about Semyon, his projects, skills, or portfolio.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        if not question:
            continue

        result = answer_question(vector_store, question)

        print("\nAssistant:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source}")

        print("\nStandalone question:")
        print(result["standalone_question"])


if __name__ == "__main__":
    main()