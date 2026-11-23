from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal, Optional


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    abstract: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    included: bool = True
    pdf_path: Optional[str] = None
    screening_reason: Optional[str] = None

class Evidence(BaseModel):
    evidence_id: str
    paper_id: str
    text: str
    doi: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    retrieval_score: Optional[float] = None


class PaperProfile(BaseModel):
    paper_id: str
    problem: str = ""
    method: str = ""
    contributions: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class ComparisonDraft(BaseModel):
    """Ce que le LLM remplit : il ne produit que des identifiants de preuves."""
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    existing_contributions: list[str] = Field(default_factory=list)
    potentially_distinct_elements: list[str] = Field(default_factory=list)
    redundancy_risks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs (e.g. E001) of the evidence passages supporting the statements. "
                    "Only IDs that appear in the input.",
    )


class Comparison(ComparisonDraft):
    """Résultat final : le contenu du brouillon, plus les vraies preuves retrouvées par le code."""
    evidence: list[Evidence] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

GapType = Literal["methodological", "technical", "empirical",
                  "evaluation", "integration", "theoretical"]


class GapDraft(BaseModel):
    """Une lacune telle que le LLM la propose (identifiants de preuves seulement)."""
    description: str
    type: GapType
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs (e.g. E001) of the evidence passages supporting this gap. "
                    "Only IDs that appear in the input.",
    )


class GapsDraft(BaseModel):
    research_gaps: list[GapDraft] = Field(default_factory=list)


class ResearchGap(BaseModel):
    """Lacune finale : preuves, articles de soutien et confiance calculés par le code."""
    description: str
    type: GapType
    supporting_papers: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class ResearchGaps(BaseModel):
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

class VerificationRequest(BaseModel):
    """Une vérification que le critique demande avant de conclure."""
    kind: Literal["search_more_literature", "check_evidence"]
    reason: str
    query: Optional[str] = Field(
        default=None,
        description="Suggested search query when kind is 'search_more_literature'.",
    )


class ReviewDraft(BaseModel):
    """Ce que le LLM remplit."""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    methodological_issues: list[str] = Field(default_factory=list)
    novelty_risks: list[str] = Field(default_factory=list)
    feasibility_level: Literal["low", "medium", "high"]
    feasibility: str = Field(
        description="One or two sentences justifying the feasibility level."
    )
    missing_evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    verification_requests: list[VerificationRequest] = Field(default_factory=list)


class CriticalReview(ReviewDraft):
    """Résultat final : le brouillon, enrichi par les contrôles du code."""
    needs_more_evidence: bool = False
    notes: list[str] = Field(default_factory=list)