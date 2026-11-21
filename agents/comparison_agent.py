from agents.common import (format_idea, format_papers, invoke_with_retries,
                           resolve_evidence)
from graph.schemas import Comparison, ComparisonDraft
from llm import structured_llm


def build_prompt(idea_profile, paper_profiles):
    return f"""You are a rigorous research analyst. Compare a research idea with the existing literature.

RESEARCH IDEA:
{format_idea(idea_profile)}

ANALYZED PAPERS (evidence passages are in brackets, e.g. [E001]):
{format_papers(paper_profiles)}

Rules:
- Use ONLY the information above. Do not rely on outside knowledge about these papers.
- Every statement must mention the paper_id(s) it relies on in square brackets, e.g. "[P1]".
- In evidence_ids, list only IDs that appear in the input. Never invent an ID.
- "potentially_distinct_elements" are parts of the idea that no analyzed paper seems to cover.
  They are candidates, NOT proven novelty.
- If the evidence is insufficient for a point, leave that list empty instead of guessing.

Fill: similarities, differences, existing_contributions, potentially_distinct_elements,
redundancy_risks, evidence_ids.
"""


def compare_research_idea(idea_profile, paper_profiles, llm=None):
    """Compare l'idée avec les articles analysés. Retourne un objet Comparison."""
    if not paper_profiles:
        return Comparison(notes=["No analyzed papers available: comparison is impossible."])

    llm = llm or structured_llm(ComparisonDraft)
    draft = invoke_with_retries(llm, build_prompt(idea_profile, paper_profiles))

    # Le code retrouve les vraies preuves ; le LLM n'a fourni que des identifiants.
    evidence, unknown = resolve_evidence(draft.evidence_ids, paper_profiles)

    notes = []
    if unknown:
        notes.append(f"Ignored unknown evidence IDs cited by the model: {unknown}")
    if not evidence:
        notes.append("No verifiable evidence cited: treat these statements as unsupported.")

    data = draft.model_dump()
    data["evidence_ids"] = [e.evidence_id for e in evidence]   # on ne garde que les valides
    return Comparison(**data, evidence=evidence, notes=notes)