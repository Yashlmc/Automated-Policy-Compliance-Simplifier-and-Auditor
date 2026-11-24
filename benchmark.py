import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from legal_analyzer import analyze_legal_document
from pdf_reader import extract_pdf_pages


LABELS = ["Low", "Medium", "High"]


def normalize_quote(text):
    """Lowercase and strip punctuation/extra whitespace for clause matching."""
    s = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(s.split())


def _token_jaccard(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def find_matching_clause(clauses, expected_quote, expected_page=None):
    """
    Match an annotated gold clause against AI-extracted clauses using:
      1. Exact normalized quote match.
      2. Substring containment (handles annotator snippet vs full clause).
      3. Token Jaccard overlap >= 0.60 on the same PDF page.
    """
    norm_exp = normalize_quote(expected_quote)
    if not norm_exp:
        return None

    # Prefer clauses on the same PDF page when provided
    page_filtered = (
        [c for c in clauses if c.pdf_page == int(expected_page)]
        if expected_page is not None and pd.notna(expected_page)
        else list(clauses)
    )
    candidates = page_filtered or list(clauses)

    # 1. Exact match
    for clause in candidates:
        if normalize_quote(clause.original_quote) == norm_exp:
            return clause

    # 2. Substring match
    for clause in candidates:
        norm_ext = normalize_quote(clause.original_quote)
        if norm_exp in norm_ext or norm_ext in norm_exp:
            return clause

    # 3. Best token Jaccard overlap >= 0.60
    best_clause = None
    best_score = 0.0
    for clause in candidates:
        score = _token_jaccard(norm_exp, normalize_quote(clause.original_quote))
        if score > best_score:
            best_score = score
            best_clause = clause

    return best_clause if best_score >= 0.60 else None


def run_benchmark(
    annotation_path="benchmark_data/annotations.csv",
    agreement_dir="benchmark_data/agreements",
    output_dir="benchmark_results",
    jurisdiction="India - Jurisdiction not specified",
    strict_count=True,
):
    """
    Evaluate AI risk classification against manually annotated rental agreements.
    Reports per-class Precision, Recall, F1, Macro averages, Confusion Matrix,
    and Clause Extraction Coverage (saving unmatched clauses explicitly).

    Returns:
        results_df, metrics_df, matrix, missing_df, coverage_pct
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(
            f"Annotations file not found: {annotation_path}"
        )

    annotations = pd.read_csv(annotation_path)

    required = [
        "agreement_file",
        "clause_id",
        "pdf_page",
        "clause_quote",
        "gold_risk",
    ]
    for column in required:
        if column not in annotations.columns:
            raise ValueError(f"Missing annotation column: {column}")

    if annotations.empty:
        raise ValueError("No annotations found in annotations.csv.")

    annotations["gold_risk"] = (
        annotations["gold_risk"].astype(str).str.strip().str.capitalize()
    )

    if not set(annotations["gold_risk"]).issubset(set(LABELS)):
        raise ValueError(
            f"Invalid gold_risk labels found. Allowed labels: {LABELS}"
        )

    agreement_files = annotations["agreement_file"].unique()

    if strict_count and not (15 <= len(agreement_files) <= 20):
        raise ValueError(
            f"Benchmark requires 15–20 distinct agreements "
            f"(found {len(agreement_files)}). Pass strict_count=False for a pilot dry-run."
        )

    predictions = []
    missing_clauses = []

    for filename in agreement_files:
        path = os.path.join(agreement_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agreement PDF not found: {path}")

        print(f"Analyzing: {filename}")
        pages = extract_pdf_pages(path)

        row_jurisdiction = jurisdiction
        if "jurisdiction" in annotations.columns:
            file_j = annotations.loc[
                annotations["agreement_file"] == filename, "jurisdiction"
            ].dropna()
            if not file_j.empty:
                row_jurisdiction = str(file_j.iloc[0])

        result = analyze_legal_document(pages, jurisdiction=row_jurisdiction)
        expected_rows = annotations[annotations["agreement_file"] == filename]

        for _, expected in expected_rows.iterrows():
            matched = find_matching_clause(
                result.clauses,
                expected["clause_quote"],
                expected_page=expected["pdf_page"],
            )

            if matched is None:
                missing_clauses.append(
                    {
                        "agreement_file": filename,
                        "clause_id": expected["clause_id"],
                        "pdf_page": expected["pdf_page"],
                        "gold_risk": expected["gold_risk"],
                        "reason": "Clause not extracted or quote mismatch",
                    }
                )
                continue

            predictions.append(
                {
                    "agreement_file": filename,
                    "clause_id": expected["clause_id"],
                    "pdf_page": expected["pdf_page"],
                    "gold_risk": expected["gold_risk"],
                    "predicted_risk": matched.risk_level,
                    "evidence_status": matched.evidence_status,
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    missing_df = pd.DataFrame(missing_clauses)
    missing_df.to_csv(
        os.path.join(output_dir, "missing_clauses.csv"), index=False
    )

    results = pd.DataFrame(predictions)
    coverage = round(len(results) / len(annotations) * 100, 2)
    print(
        f"\nClause extraction coverage: {coverage:.2f}% "
        f"({len(results)}/{len(annotations)} matched; "
        f"{len(missing_clauses)} missing)"
    )

    if results.empty:
        raise ValueError("No annotated clauses matched AI predictions.")

    y_true = results["gold_risk"]
    y_pred = results["predicted_risk"]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
    )
    print("CONFUSION MATRIX (Rows=Actual, Cols=Predicted [Low, Medium, High])")
    print(matrix)
    print(
        f"MACRO PRECISION: {precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}"
    )
    print(
        f"MACRO RECALL:    {recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}"
    )
    print(
        f"MACRO F1:        {f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}"
    )

    results.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    metrics_df = pd.DataFrame(report).transpose().round(4)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Actual Risk")
    ax.set_title(
        f"Legal Risk Classification (Coverage: {coverage:.1f}%)"
    )

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"\nBenchmark results saved in {output_dir}/")
    return results, metrics_df, matrix, missing_df, coverage


if __name__ == "__main__":
    run_benchmark()