from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ==================================================
# 1. RENTAL AGREEMENT SIMPLIFIER SCHEMAS
# ==================================================

RiskLevel = Literal["Low", "Medium", "High"]

EvidenceStatus = Literal[
    "Supported by retrieved reference",
    "Potential concern",
    "Insufficient legal evidence",
]


class LegalCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)
    provision: str = Field(min_length=1)
    source_page: int = Field(ge=1)
    supporting_quote: str = Field(min_length=1)


class LegalClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_title: str = Field(min_length=1)
    original_quote: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    simple_meaning: str = Field(min_length=1)
    tenant_obligations: List[str] = Field(default_factory=list)
    landlord_obligations: List[str] = Field(default_factory=list)
    risk_level: RiskLevel
    risk_reason: str = Field(min_length=1)
    legal_references: List[LegalCitation] = Field(default_factory=list)
    evidence_status: EvidenceStatus


class LegalDocumentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_summary: str = Field(min_length=1)
    jurisdiction_assumption: str = Field(min_length=1)
    clauses: List[LegalClause] = Field(default_factory=list)
    overall_observations: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "This AI-generated analysis is for informational purposes only "
        "and does not constitute legal advice."
    )


# ==================================================
# 2. POLICY & COMPLIANCE AUDITOR SCHEMAS
# ==================================================

ComplianceStatus = Literal["Compliant", "Partial", "Missing"]


class ComplianceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standard_provision: str = Field(min_length=1)
    requirement_summary: str = Field(min_length=1)
    compliance_status: ComplianceStatus
    policy_quote: Optional[str] = None
    policy_page: Optional[int] = Field(default=None, ge=1)
    standard_citations: List[LegalCitation] = Field(default_factory=list)
    gap_explanation: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    escalation_reason: Optional[str] = None
    injection_detected: bool = False


class ComplianceAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standard_name: str = Field(min_length=1)
    policy_summary: str = Field(min_length=1)
    items: List[ComplianceItem] = Field(default_factory=list)
    high_priority_gaps: List[str] = Field(default_factory=list)
    injection_alerts: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "This automated compliance audit is for preliminary review only "
        "and does not constitute formal legal or regulatory certification."
    )


# ==================================================
# 3. LLM-AS-A-JUDGE EVALUATION SCHEMA
# ==================================================

class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness_score: int = Field(ge=1, le=5)
    citation_precision_score: int = Field(ge=1, le=5)
    unsupported_claims: List[str] = Field(default_factory=list)
    verdict_rationale: str = Field(min_length=1)