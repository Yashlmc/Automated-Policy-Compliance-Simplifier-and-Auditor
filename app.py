import json

import pandas as pd
import streamlit as st

from compliance_auditor import audit_policy_document
from evaluator import (
    evaluate_audit_with_judge,
    run_judge_calibration,
    run_prompt_injection_redteam,
    run_retrieval_benchmark,
)
from legal_analyzer import analyze_legal_document, get_legal_corpus
from pdf_reader import extract_pdf_pages
from rag_engine import list_indexed_sources


st.set_page_config(
    page_title="Legal Simplifier & Compliance Auditor",
    page_icon="⚖️",
    layout="wide",
)


# ==========================================
# HEADER & DISCLAIMER
# ==========================================

st.title("⚖️ Legal Document Simplifier & Compliance Auditor")

st.markdown(
    """
    Unified legal & regulatory AI workbench combining **Hybrid RAG**
    (BM25 + FAISS Dense + Cross-Encoder Reranking), **Pydantic structured
    verification**, **LLM-as-a-Judge evaluation**, and **Indirect Prompt-Injection
    Red-Teaming**.
    """
)

st.warning(
    """
    **Legal Disclaimer — Not Legal Advice**

    Outputs are AI-generated preliminary review indicators, not binding legal
    advice or regulatory certification. Always verify applicable state/central
    legislation (e.g., Maharashtra Rent Control Act, 1999; DPDP Act, 2023)
    and consult qualified legal counsel.
    """
)


# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Rental Simplifier",
        "🛡️ Policy Compliance Auditor",
        "📊 Benchmarks & Judge Calibration",
        "🧪 Prompt-Injection Red-Team Lab",
    ]
)


# ==========================================
# TAB 1: RENTAL AGREEMENT SIMPLIFIER
# ==========================================

with tab1:
    st.subheader("Upload Rental Agreement")

    c1, c2 = st.columns([2, 1])
    with c1:
        jurisdiction = st.selectbox(
            "Select agreement jurisdiction",
            [
                "India - Jurisdiction not specified",
                "Maharashtra, India",
                "Other Indian state",
                "Other country",
            ],
        )
    with c2:
        simplifier_mode = st.selectbox(
            "Retrieval mode",
            ["hybrid", "dense", "bm25"],
            index=0,
            key="simplifier_mode",
        )

    uploaded_file = st.file_uploader(
        "Upload rental agreement PDF", type=["pdf"], key="rental_pdf"
    )

    if uploaded_file is not None:
        file_sig = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("legal_file_sig") != file_sig:
            for key in ["legal_pages", "legal_result", "legal_filename"]:
                st.session_state.pop(key, None)
            st.session_state["legal_file_sig"] = file_sig

        if st.button(
            "🔍 Analyze Legal Agreement",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Extracting PDF pages, running Hybrid RAG, and verifying quotes..."
                ):
                    pages = extract_pdf_pages(uploaded_file)
                    result = analyze_legal_document(
                        pages,
                        jurisdiction=jurisdiction,
                        retrieval_mode=simplifier_mode,
                    )
                    st.session_state["legal_pages"] = pages
                    st.session_state["legal_result"] = result.model_dump()
                    st.session_state["legal_filename"] = uploaded_file.name
                    st.session_state["legal_jurisdiction"] = jurisdiction

                st.success("Document analysis and quote verification completed!")
            except Exception as e:
                st.error(f"Analysis or verification failed: {e}")

    if "legal_result" in st.session_state:
        result = st.session_state["legal_result"]
        clauses = result.get("clauses", [])

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Clauses", len(clauses))
        m2.metric("High Risk", sum(c["risk_level"] == "High" for c in clauses))
        m3.metric("Medium Risk", sum(c["risk_level"] == "Medium" for c in clauses))
        m4.metric("Low Risk", sum(c["risk_level"] == "Low" for c in clauses))

        st.write(result["document_summary"])
        st.caption(f"Jurisdiction note: {result['jurisdiction_assumption']}")

        table_df = pd.DataFrame(
            [
                {
                    "Clause": c["clause_title"],
                    "Page": c["pdf_page"],
                    "Risk": c["risk_level"],
                    "Meaning": c["simple_meaning"],
                    "Evidence Status": c["evidence_status"],
                }
                for c in clauses
            ]
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        risk_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        for idx, clause in enumerate(clauses, start=1):
            icon = risk_icons.get(clause["risk_level"], "🟡")
            with st.expander(
                f"{icon} {idx}. {clause['clause_title']} (Page {clause['pdf_page']})"
            ):
                st.markdown("**Original PDF quotation (verified)**")
                st.info(clause["original_quote"])
                st.markdown(f"**Simple meaning:** {clause['simple_meaning']}")
                st.markdown(
                    f"**Risk ({clause['risk_level']}):** {clause['risk_reason']}"
                )
                if clause["legal_references"]:
                    for ref in clause["legal_references"]:
                        st.markdown(
                            f"**{ref['source_name']}** — {ref['provision']} (p. {ref['source_page']})"
                        )
                        st.info(ref["supporting_quote"])

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download JSON Report",
            data=json.dumps(result, indent=4, ensure_ascii=False),
            file_name="legal_analysis.json",
            mime="application/json",
            use_container_width=True,
        )
        d2.download_button(
            "Download Clause CSV",
            data=table_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="legal_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==========================================
# TAB 2: POLICY & COMPLIANCE AUDITOR
# ==========================================

with tab2:
    st.subheader("Automated Policy & Compliance Auditor (Hybrid RAG + LLM-as-a-Judge)")

    try:
        corpus_bundle = get_legal_corpus()
        available_standards = ["All Indexed Standards"] + list_indexed_sources(
            corpus_bundle
        )
    except Exception as e:
        available_standards = ["All Indexed Standards"]
        st.info(
            f"Corpus not loaded yet ({e}). Place DPDP/ISO/Tenancy PDFs in "
            "legal_corpus/ and run `python ingest_corpus.py`."
        )

    a1, a2, a3 = st.columns([2, 1, 1])
    with a1:
        selected_standard = st.selectbox(
            "Target Regulatory Standard", available_standards
        )
    with a2:
        audit_mode = st.selectbox(
            "Retriever", ["hybrid", "dense", "bm25"], index=0, key="audit_mode"
        )
    with a3:
        conf_threshold = st.slider(
            "Human-Escalation Confidence Cutoff", 0.50, 0.95, 0.75, 0.05
        )

    run_judge_toggle = st.checkbox(
        "Run automated LLM-as-a-Judge faithfulness & citation scoring",
        value=True,
    )

    policy_pdf = st.file_uploader(
        "Upload target organization's policy draft (PDF)",
        type=["pdf"],
        key="policy_pdf",
    )

    if policy_pdf is not None:
        p_sig = (policy_pdf.name, policy_pdf.size, selected_standard)
        if st.session_state.get("policy_sig") != p_sig:
            for k in ["audit_report", "audit_judge_df", "audit_context"]:
                st.session_state.pop(k, None)
            st.session_state["policy_sig"] = p_sig

        if st.button(
            "🛡️ Run Compliance Audit", type="primary", use_container_width=True
        ):
            try:
                with st.spinner(
                    "Running Hybrid RAG compliance audit & indirect prompt-injection checks..."
                ):
                    p_pages = extract_pdf_pages(policy_pdf)
                    audit_report, std_context = audit_policy_document(
                        pages=p_pages,
                        standard_name=selected_standard,
                        confidence_threshold=conf_threshold,
                        retrieval_mode=audit_mode,
                    )

                judge_df = None
                if run_judge_toggle:
                    with st.spinner(
                        "Scoring faithfulness & citation precision with LLM-as-a-Judge..."
                    ):
                        judge_df = evaluate_audit_with_judge(
                            audit_report, p_pages, std_context
                        )

                st.session_state["audit_report"] = audit_report.model_dump()
                st.session_state["audit_judge_df"] = judge_df
                st.success("Compliance audit completed and verified!")

            except Exception as e:
                st.error(f"Compliance audit failed: {e}")

    if "audit_report" in st.session_state:
        audit = st.session_state["audit_report"]
        items = audit.get("items", [])

        if audit.get("injection_alerts"):
            st.error(
                "⚠️ Indirect Prompt-Injection Alert(s) Detected in Uploaded Policy:\n- "
                + "\n- ".join(audit["injection_alerts"])
            )

        st.divider()
        c_ok = sum(i["compliance_status"] == "Compliant" for i in items)
        c_part = sum(i["compliance_status"] == "Partial" for i in items)
        c_miss = sum(i["compliance_status"] == "Missing" for i in items)
        c_esc = sum(i["needs_human_review"] for i in items)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Compliant", c_ok)
        k2.metric("Partial", c_part)
        k3.metric("Missing", c_miss)
        k4.metric("Human Escalations", c_esc)

        st.subheader("Clause-by-Clause Compliance Matrix")
        matrix_df = pd.DataFrame(
            [
                {
                    "Provision": i["standard_provision"],
                    "Requirement": i["requirement_summary"],
                    "Status": i["compliance_status"],
                    "Policy Page": i["policy_page"] or "—",
                    "Confidence": round(i["confidence_score"], 2),
                    "Human Review": "YES" if i["needs_human_review"] else "No",
                    "Gap / Notes": i["gap_explanation"],
                }
                for i in items
            ]
        )
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        # Confidence-Based Human-Escalation Queue
        escalated = [i for i in items if i["needs_human_review"]]
        st.subheader(f"🚨 Human-Escalation Queue ({len(escalated)} items)")
        if escalated:
            esc_df = pd.DataFrame(
                [
                    {
                        "Provision": i["standard_provision"],
                        "Status": i["compliance_status"],
                        "Confidence": round(i["confidence_score"], 2),
                        "Escalation Reason": i["escalation_reason"],
                    }
                    for i in escalated
                ]
            )
            st.dataframe(esc_df, use_container_width=True, hide_index=True)
        else:
            st.success("No items triggered the human-escalation threshold.")

        if st.session_state.get("audit_judge_df") is not None:
            st.subheader("⚖️ LLM-as-a-Judge Scores (Faithfulness & Citation Precision)")
            st.dataframe(
                st.session_state["audit_judge_df"],
                use_container_width=True,
                hide_index=True,
            )

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Download Full Audit Report (JSON)",
            data=json.dumps(audit, indent=4, ensure_ascii=False),
            file_name="compliance_audit_report.json",
            mime="application/json",
            use_container_width=True,
        )
        dl2.download_button(
            "Download Compliance Matrix (CSV)",
            data=matrix_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="compliance_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==========================================
# TAB 3: BENCHMARKS & JUDGE CALIBRATION
# ==========================================

with tab3:
    st.subheader("1. Hybrid vs. Dense Retrieval Benchmark (Recall@5 & MRR@5)")
    st.write(
        "Proves why BM25 + FAISS RRF + Cross-Encoder outperforms pure vector search "
        "on exact statutory clause numbers (`benchmark_data/retrieval_queries.csv`)."
    )
    if st.button("Run Hybrid vs. Dense Benchmark", use_container_width=True):
        try:
            with st.spinner("Evaluating Dense, BM25, and Hybrid retrievers..."):
                ret_sum, ret_det = run_retrieval_benchmark(top_k=5)
                st.session_state["ret_sum"] = ret_sum
                st.session_state["ret_det"] = ret_det
            st.success("Retrieval benchmark completed!")
        except Exception as e:
            st.error(f"Retrieval benchmark failed: {e}")

    if "ret_sum" in st.session_state:
        st.dataframe(
            st.session_state["ret_sum"], use_container_width=True, hide_index=True
        )
        with st.expander("View per-query retrieval hits"):
            st.dataframe(
                st.session_state["ret_det"], use_container_width=True, hide_index=True
            )

    st.divider()
    st.subheader("2. LLM-as-a-Judge vs. Human Agreement (15+ Graded Clauses)")
    st.write(
        "Computes Quadratic Weighted Cohen's Kappa, Spearman correlation, MAE, and "
        "within-±1 agreement from `benchmark_data/judge_calibration.csv`."
    )
    if st.button("Run Judge-vs-Human Calibration Report", use_container_width=True):
        try:
            cal_sum, cal_raw = run_judge_calibration()
            st.session_state["cal_sum"] = cal_sum
            st.session_state["cal_raw"] = cal_raw
            st.success("Judge calibration report computed!")
        except Exception as e:
            st.error(f"Calibration failed: {e}")

    if "cal_sum" in st.session_state:
        st.dataframe(
            st.session_state["cal_sum"], use_container_width=True, hide_index=True
        )

    st.divider()
    st.subheader("3. Rental Agreement Risk Classification Benchmark (15–20 Agreements)")
    strict_count = st.checkbox(
        "Enforce 15–20 distinct agreements (uncheck for pilot dry-run)", value=True
    )
    if st.button("Run Risk Classification Benchmark", use_container_width=True):
        try:
            from benchmark import run_benchmark

            with st.spinner("Running risk benchmark across annotated agreements..."):
                results, metrics, matrix, missing_df, coverage = run_benchmark(
                    strict_count=strict_count
                )
                st.session_state["bench_results"] = results
                st.session_state["bench_metrics"] = metrics
                st.session_state["bench_matrix"] = matrix
                st.session_state["bench_missing"] = missing_df
                st.session_state["bench_coverage"] = coverage
            st.success("Risk benchmark completed!")
        except Exception as e:
            st.error(f"Risk benchmark failed: {e}")

    if "bench_metrics" in st.session_state:
        st.metric(
            "Clause Extraction Coverage",
            f"{st.session_state['bench_coverage']:.2f}%",
        )
        st.dataframe(st.session_state["bench_metrics"], use_container_width=True)


# ==========================================
# TAB 4: INDIRECT PROMPT-INJECTION RED-TEAM LAB
# ==========================================

with tab4:
    st.subheader("🧪 Indirect Prompt-Injection Red-Team Suite")
    st.write(
        "Tests whether untrusted directives hidden inside an uploaded policy draft "
        "can hijack the compliance auditor's verdicts or confidence scores."
    )

    rt_standard = st.selectbox(
        "Standard to test against", available_standards, key="rt_standard"
    )

    if st.button("Run Red-Team Security Suite", type="primary", use_container_width=True):
        try:
            with st.spinner("Running 4 adversarial & control policy payloads..."):
                rt_df, block_rate = run_prompt_injection_redteam(
                    standard_name=rt_standard
                )
                st.session_state["rt_df"] = rt_df
                st.session_state["rt_block_rate"] = block_rate
            st.success("Red-team evaluation completed!")
        except Exception as e:
            st.error(f"Red-team run failed: {e}")

    if "rt_df" in st.session_state:
        st.metric(
            "Indirect Prompt-Injection Block Rate",
            f"{st.session_state['rt_block_rate']:.1f}%",
        )
        st.dataframe(
            st.session_state["rt_df"], use_container_width=True, hide_index=True
        )
        st.download_button(
            "Download Red-Team Report (CSV)",
            data=st.session_state["rt_df"].to_csv(index=False).encode("utf-8-sig"),
            file_name="redteam_injection_report.csv",
            mime="text/csv",
        )


st.divider()
st.caption(
    "Legal Document Simplifier & Compliance Auditor | Hybrid RAG (BM25 + FAISS + "
    "Cross-Encoder) | Pydantic | LLM-as-a-Judge | Groq | Streamlit"
)