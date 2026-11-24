import json
import os
import re

import faiss
import fitz
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


CORPUS_DIR = "legal_corpus"
STORE_DIR = "vector_store"

INDEX_PATH = os.path.join(STORE_DIR, "legal.index")
METADATA_PATH = os.path.join(STORE_DIR, "metadata.json")

EMBEDDING_MODEL = os.getenv(
    "LEGAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANKER_MODEL = os.getenv(
    "LEGAL_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


def normalize_text(text):
    """Collapse repeated whitespace into single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_legal(text):
    """
    Tokenize legal text while preserving exact statutory clause numbers
    such as 'section', '8(7)', '11(1)', '4.2.1', 'iso-27001', 'dpdp'.
    """
    return re.findall(r"[a-z0-9]+(?:[.\-/(][a-z0-9]+\)?)*", str(text).lower())


def extract_provision(text):
    """
    Extract statutory section/article/clause/ISO identifiers from regulatory text.
    """
    patterns = [
        r"(?:Section|Article|Clause|Rule|Sub-section|Chapter|Schedule)\s+\d+[A-Za-z]?(?:\(\d+\))*",
        r"\b(?:ISO|IEC|DPDP|Clause)\s*\d+(?:\.\d+)+",
        r"(?m)^\s*(\d+[A-Za-z]?)\.\s+[A-Z][A-Za-z\s,-]{3,40}",
    ]

    for pat in patterns[:2]:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    bare_match = re.search(patterns[2], text)
    if bare_match:
        return f"Section {bare_match.group(1)}"

    return "Provision not identified"


def split_into_chunks(text, max_words=180, overlap=30):
    """Split page text into overlapping word chunks to preserve clause boundaries."""
    words = text.split()
    if not words:
        return []

    step = max(1, max_words - overlap)
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words]).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        if i + max_words >= len(words):
            break

    return chunks


def build_corpus(corpus_dir=CORPUS_DIR, store_dir=STORE_DIR):
    """
    Read all regulatory & legal PDFs in legal_corpus/ (e.g., Model Tenancy Act,
    Maharashtra Rent Control Act, DPDP Act 2023, ISO/Privacy standards),
    chunk with page provenance, embed with Sentence Transformers, and save FAISS + metadata.
    """
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(
            f"Corpus directory '{corpus_dir}' not found. "
            "Create it and add your legal/regulatory PDFs first."
        )

    os.makedirs(store_dir, exist_ok=True)
    documents = []

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(corpus_dir, filename)
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf):
                page_text = page.get_text("text")
                if not page_text or not page_text.strip():
                    continue

                chunks = split_into_chunks(page_text)
                for chunk in chunks:
                    documents.append(
                        {
                            "text": chunk,
                            "source_name": filename,
                            "provision": extract_provision(chunk),
                            "source_page": page_index + 1,
                        }
                    )

    if not documents:
        raise ValueError(
            "No readable legal PDF content found in legal_corpus/. "
            "Check that your PDFs contain selectable text."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [doc["text"] for doc in documents]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = os.path.join(store_dir, "legal.index")
    metadata_path = os.path.join(store_dir, "metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)

    return len(documents)


def load_corpus(store_dir=STORE_DIR, load_reranker=True):
    """
    Load persisted FAISS index, chunk metadata, BM25 lexical index,
    embedding model, and optional Cross-Encoder reranker.

    Returns a dictionary bundle used by dense, BM25, and hybrid retrievers.
    """
    index_path = os.path.join(store_dir, "legal.index")
    metadata_path = os.path.join(store_dir, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Legal vector index not found. Run 'python ingest_corpus.py' first."
        )

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    tokenized_corpus = [
        tokenize_legal(
            f"{doc['source_name']} {doc['provision']} {doc['text']}"
        )
        for doc in metadata
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL) if load_reranker else None

    return {
        "embed_model": embed_model,
        "faiss_index": index,
        "bm25": bm25,
        "metadata": metadata,
        "reranker": reranker,
    }


def list_indexed_sources(corpus):
    """Return sorted list of unique PDF source filenames in the vector store."""
    return sorted({doc["source_name"] for doc in corpus["metadata"]})


def reciprocal_rank_fusion(rank_lists, k=60):
    """
    Fuse multiple ranked lists of document indices using Reciprocal Rank Fusion.
    RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    fused_scores = {}
    for ranked_indices in rank_lists:
        for rank, doc_idx in enumerate(ranked_indices, start=1):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + (
                1.0 / (k + rank)
            )
    return sorted(
        fused_scores.items(), key=lambda pair: pair[1], reverse=True
    )


def retrieve_legal_context(
    query,
    corpus,
    top_k=5,
    mode="hybrid",
    source_filter=None,
    jurisdiction=None,
):
    """
    Retrieve relevant legal/regulatory chunks using:
      - mode="dense"  : Pure FAISS cosine similarity
      - mode="bm25"   : Pure BM25 lexical matching (excels on exact Section numbers)
      - mode="hybrid" : BM25 + FAISS fused via RRF + Cross-Encoder reranking

    Optional source_filter restricts candidates to a specific PDF filename
    (e.g. 'dpdp_act_2023.pdf').
    """
    metadata = corpus["metadata"]
    if not query or not query.strip() or not metadata:
        return []

    if source_filter and source_filter != "All Indexed Standards":
        allowed_ids = {
            i
            for i, doc in enumerate(metadata)
            if doc["source_name"] == source_filter
        }
    else:
        allowed_ids = set(range(len(metadata)))

    if not allowed_ids:
        return []

    candidate_pool = min(max(top_k * 4, 20), len(metadata))

    # 1. Dense retrieval (FAISS)
    dense_ranked = []
    dense_scores = {}
    if mode in {"dense", "hybrid"}:
        q_emb = corpus["embed_model"].encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, indices = corpus["faiss_index"].search(q_emb, candidate_pool)
        for score, idx in zip(scores[0], indices[0]):
            if idx in allowed_ids:
                dense_ranked.append(int(idx))
                dense_scores[int(idx)] = float(score)

    # 2. Lexical retrieval (BM25)
    bm25_ranked = []
    bm25_scores = {}
    if mode in {"bm25", "hybrid"}:
        q_tokens = tokenize_legal(query)
        raw_bm25 = corpus["bm25"].get_scores(q_tokens)
        scored_bm25 = [
            (idx, float(raw_bm25[idx]))
            for idx in allowed_ids
            if raw_bm25[idx] > 0
        ]
        scored_bm25.sort(key=lambda x: x[1], reverse=True)
        for idx, sc in scored_bm25[:candidate_pool]:
            bm25_ranked.append(idx)
            bm25_scores[idx] = sc

    # 3. Combine & Rank
    if mode == "dense":
        selected_ids = dense_ranked[:top_k]
        score_map = dense_scores
    elif mode == "bm25":
        selected_ids = bm25_ranked[:top_k]
        score_map = bm25_scores
    else:
        # Hybrid RRF fusion
        rrf_results = reciprocal_rank_fusion([dense_ranked, bm25_ranked])
        rrf_ids = [idx for idx, _ in rrf_results[: max(top_k * 3, 15)]]
        score_map = dict(rrf_results)

        # Cross-Encoder Reranking
        reranker = corpus.get("reranker")
        if reranker is not None and rrf_ids:
            pairs = [
                (query, f"{metadata[i]['provision']}: {metadata[i]['text']}")
                for i in rrf_ids
            ]
            ce_scores = reranker.predict(pairs)
            reranked = sorted(
                zip(rrf_ids, [float(s) for s in ce_scores]),
                key=lambda x: x[1],
                reverse=True,
            )
            selected_ids = [idx for idx, _ in reranked[:top_k]]
            score_map = dict(reranked)
        else:
            selected_ids = rrf_ids[:top_k]

    is_maharashtra = (
        isinstance(jurisdiction, str) and "maharashtra" in jurisdiction.lower()
    )

    results = []
    for idx in selected_ids:
        item = metadata[idx].copy()
        sim = float(score_map.get(idx, 0.0))
        if is_maharashtra and "maharashtra" in item["source_name"].lower():
            sim += 0.05
        item["similarity"] = sim
        item["retrieval_mode"] = mode
        results.append(item)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]