from rag_engine import build_corpus


if __name__ == "__main__":
    print("Building legal reference corpus from legal_corpus/ ...")
    count = build_corpus()
    print(f"Successfully indexed {count} legal text chunks.")
    print("FAISS index and metadata saved in vector_store/.")