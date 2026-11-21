from typing import Optional
from pydantic import BaseModel, Field


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