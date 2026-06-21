import shutil

from app.rag.vectorstore import PERSIST_DIR, build_vectorstore


def main():
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print(f"Removed existing vector store: {PERSIST_DIR}")

    vector_store = build_vectorstore()
    count = vector_store._collection.count()
    print(f"Vector store created successfully with {count} chunks.")


if __name__ == "__main__":
    main()
