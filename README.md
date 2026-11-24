# Legal Document Simplifier & Automated Policy Compliance Auditor

An end-to-end Generative AI legal and regulatory analysis workbench built with **Hybrid RAG** (BM25 + FAISS Dense Embeddings + Cross-Encoder Reranking), **Pydantic v2** structured schema enforcement, **bare-act resilient quotation verification**, **indirect prompt-injection defenses**, and an **LLM-as-a-Judge** evaluation harness.

> **Legal Disclaimer:** This application generates preliminary, informational AI analysis only. It does not constitute legal advice, verify contract enforceability, or provide regulatory certification. The Indian Model Tenancy Act, 2021 is a model framework circulated for state adoption and is not automatically binding law across all jurisdictions; always verify applicable state legislation (e.g., the Maharashtra Rent Control Act, 1999) and consult qualified legal counsel.

## Key Capabilities

* **Rental Agreement Simplifier (Tab 1):** Extracts page-numbered clauses from uploaded rental agreement PDFs via PyMuPDF, retrieves relevant statutory provisions using jurisdiction-aware Hybrid RAG, explains each clause in plain English, separates tenant and landlord obligations, assigns `Low` / `Medium` / `High` review-priority risk levels, and verifies every quotation against the source PDF page.

* **Automated Policy & Compliance Auditor (Tab 2):** Audits an organization's policy draft PDF against an indexed regulatory standard (e.g., India's DPDP Act, 2023 or ISO/privacy standards), outputs a structured `Compliant` / `Partial` / `Missing` matrix with verbatim evidence and page numbers, scores every item with an automated LLM-as-a-Judge (`Faithfulness` and `Citation Precision` on a 1–5 scale), and routes uncertain or missing items to a **Confidence-Based Human-Escalation Queue**.

* **Benchmarks & Judge Calibration (Tab 3):** Runs three reproducible evaluation suites inside the UI or CLI:

  * **Hybrid vs. Dense vs. BM25 Retrieval Benchmark** (`Recall@5` and `MRR@5` on exact statutory clause queries).

  * **LLM-as-a-Judge vs. Human Calibration** (Quadratic Weighted Cohen's Kappa, Spearman ρ, MAE, Exact and Within-±1 Agreement across 15+ human-graded clauses).

  * **Rental Risk Classification Benchmark** (per-class Precision, Recall, Macro F1, Confusion Matrix, and Clause Extraction Coverage across 15–20 annotated agreements).

* **Indirect Prompt-Injection Red-Team Lab (Tab 4):** Stress-tests the compliance auditor against adversarial directives embedded inside untrusted policy text (`Direct Verdict Override`, `Role Hijack`, `XML Escape Attempt`, and a `Benign Control`) and reports the Attack Block Rate.

## Architecture & Repository Structure

| File / Directory | Layer | Responsibility | 
| ----- | ----- | ----- | 
| `app.py` | Presentation | 4-tab Streamlit workbench, dataset-signature session-state resets, and UTF-8-SIG CSV / JSON exports. | 
| `pdf_reader.py` | Ingestion | PyMuPDF (`fitz`) page-aware text extraction with stream `seek(0)` safety and direct path/bytes support. | 
| `ingest_corpus.py` | Indexing CLI | One-time script to chunk, embed, and persist all regulatory PDFs in `legal_corpus/` into `vector_store/`. | 
| `rag_engine.py` | Hybrid RAG | 180-word overlapping chunking (30-word overlap), bare-act section regex, FAISS `IndexFlatIP`, `BM25Okapi` legal tokenizer, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking. | 
| `schemas.py` | Data Contracts | Strict Pydantic v2 models (`ConfigDict(extra="forbid")`) for rental analysis, compliance audits, and judge scores. | 
| `legal_analyzer.py` | TA2 Simplifier | Lazy Groq/FAISS caching, jurisdiction-aware prompting, PDF hyphen/ellipsis-safe quote verification, and graceful citation sanitization. | 
| `compliance_auditor.py` | Final Jury Auditor | Regex prompt-injection pre-scanner, `<UNTRUSTED_POLICY_DOCUMENT>` XML sandboxing, compliance matrix generation, and escalation routing. | 
| `evaluator.py` | Evaluation & Red-Team | LLM-as-a-Judge grader, 15-clause human calibration metrics, Hybrid-vs-Dense retrieval benchmark, and red-team security suite. | 
| `benchmark.py` | TA2 Benchmark | Multi-agreement risk classification evaluation, confusion matrix plotting, and `missing_clauses.csv` coverage logging. | 
| `legal_corpus/` | Reference Data | Place regulatory PDFs here (e.g., `model_tenancy_act_2021.pdf`, `maharashtra_rent_control_act.pdf`, `dpdp_act_2023.pdf`). | 
| `vector_store/` | Persisted Index | Generated `legal.index` (FAISS) and `metadata.json` (chunk provenance). | 
| `benchmark_data/` | Evaluation Data | Holds `annotations.csv`, `agreements/`, `retrieval_queries.csv`, and `judge_calibration.csv`. | 

## Technology Stack

| Component | Technology | Role in System | 
| ----- | ----- | ----- | 
| Dense Vector Search | FAISS (`faiss-cpu`, `IndexFlatIP`) | Exact inner-product cosine similarity over L2-normalized embeddings with local disk persistence. | 
| Semantic Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) | 384-dimensional bi-encoder embeddings for semantic paraphrase recall across clauses and statutes. | 
| Lexical Search | `rank-bm25` (`BM25Okapi`) | Exact-term matching with a custom legal tokenizer preserving identifiers like `Section 8(7)` and `ISO 5.1.2`. | 
| Fusion & Reranking | RRF (`k=60`) + Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Scale-invariant rank fusion followed by full cross-attention scoring on `(query, provision + text)` pairs. | 
| PDF Processing | PyMuPDF (`fitz`) | Fast text extraction preserving 1-based page provenance and handling bare-act line breaks. | 
| LLM Inference | Groq API (`openai/gpt-oss-120b`) | Deterministic JSON-mode generation (`temperature=0`) via low-latency API calls. | 
| Schema Enforcement | Pydantic v2 (`pydantic>=2.7`) | Enforces strict JSON keys, `Literal` risk/compliance enums, page bounds, and confidence ranges. | 
| Evaluation Metrics | Scikit-Learn, SciPy, Matplotlib | Precision, Recall, F1, Confusion Matrix, Quadratic Weighted Cohen's Kappa, and Spearman ρ. | 
| Web Interface | Streamlit & Pandas | Interactive 4-tab dashboard with live filters, escalation queue views, and downloadable reports. | 

## Quickstart

### 1. Prerequisites

* Python 3.10 – 3.12 (recommended on Windows for prebuilt `faiss-cpu` wheels)

* A Groq API key (`GROQ_API_KEY`)

### 2. Installation

```
git clone <your-repository-url>
cd Legal-Document-Simplifier
python -m pip install -r requirements.txt

```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_actual_api_key
GROQ_MODEL=openai/gpt-oss-120b

```

### 4. Ingest Regulatory Reference PDFs

Place selectable-text legal/regulatory PDFs inside `legal_corpus/` (for example, the Model Tenancy Act 2021, Maharashtra Rent Control Act 1999, and DPDP Act 2023), then build the FAISS index once:

```
mkdir legal_corpus
# Copy your reference PDFs into legal_corpus/
python ingest_corpus.py

```

### 5. Launch the Application

```
python -m streamlit run app.py

```

## Benchmark & Calibration Data Formats

To run the automated evaluation suites in Tab 3 (or via `python benchmark.py`), populate `benchmark_data/` using the schemas below:

* **`benchmark_data/annotations.csv`** (paired with 15–20 PDFs in `benchmark_data/agreements/`):
  `agreement_file,clause_id,pdf_page,clause_quote,gold_risk`

* **`benchmark_data/retrieval_queries.csv`**:
  `query,expected_provision,expected_source`

* **`benchmark_data/judge_calibration.csv`** (minimum 15 human-graded rows, scores 1–5):
  `clause_id,human_faithfulness,judge_faithfulness,human_citation,judge_citation`

## Security & Verification Safeguards

* **Bare-Act Resilient Quote Grounding:** Rejoins PDF line-wrap hyphenation (e.g., `secu-\nrity` → `security`), validates ellipsis-abbreviated excerpts, auto-corrects adjacent page citations, and removes any unverified statutory citation rather than displaying fabricated references.

* **Two-Layer Indirect Prompt-Injection Defense:** Runs a deterministic regex pre-scanner on uploaded policy PDFs and wraps raw document text inside `<UNTRUSTED_POLICY_DOCUMENT>` XML delimiters with explicit system-level data-vs-instruction isolation.

* **Human-Escalation Gate:** Automatically routes any audited provision with `confidence_score < 0.75`, `compliance_status == "Missing"`, `injection_detected == True`, or an LLM-as-a-Judge score `< 4` into the UI Human-Escalation Queue.

## Known Limitations & Roadmap

* **Scanned PDFs:** Image-only PDFs require an OCR pre-processing step (e.g., Tesseract or PaddleOCR) before extraction.

* **Production PII Redaction:** Real rental agreements contain personal identifiers (Aadhaar, PAN, addresses); integrate an automated PII redaction layer (such as Microsoft Presidio) before sending confidential documents to external LLM APIs.

* **Future Enhancements:** Section-AST bare-act chunking, multilingual Marathi/Hindi explanation output, and automated OCR fallback.