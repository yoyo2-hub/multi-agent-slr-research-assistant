import re

from agents.common import (format_idea, format_papers, invoke_with_retries,
                           resolve_evidence)
from graph.schemas import Comparison, ComparisonDraft
from llm import structured_llm

MIN_PAPERS_FOR_NOVELTY = 5
BRACKET = re.compile(r"\[([^\]]+)\]")


def build_prompt(idea_profile, paper_profiles):
    return f"""You are a rigorous research analyst. Compare a research idea with the existing literature.

RESEARCH IDEA:
{format_idea(idea_profile)}

ANALYZED PAPERS (evidence passages are in brackets, e.g. [E001]):
{format_papers(paper_profiles)}

Rules:
- Use ONLY the information above. Do not rely on outside knowledge about these papers.
- Every statement must cite the paper_id(s) it relies on in square brackets, e.g. "[P1]".
- A statement about what a paper does NOT do or does NOT cover is allowed only if a limitation
  or evidence passage above says so. Otherwise do not state it: missing information in a summary
  is not proof of absence.
- "redundancy_risks" are places where an existing paper ALREADY does something close to the idea.
  A paper's limitation is not a redundancy risk.
- "potentially_distinct_elements": cite the paper(s) whose limitation or scope motivates the element.
  If no analyzed paper addresses it at all, end the statement with "(not covered by the analyzed papers)".
  These are candidates, NOT proven novelty.
- Do not repeat the same point in several lists.
- In evidence_ids, list only IDs that appear in the input. Never invent an ID.
- If the evidence is insufficient for a point, leave that list empty instead of guessing.

Fill: similarities, differences, existing_contributions, potentially_distinct_elements,
redundancy_risks, evidence_ids.
"""


def find_unknown_citations(draft, paper_profiles):
    """Identifiants cités entre crochets dans les phrases mais absents des profils."""
    known = {p["paper_id"] for p in paper_profiles}
    known |= {ev["evidence_id"] for p in paper_profiles for ev in p.get("evidence", [])}

    statements = (draft.similarities + draft.differences + draft.existing_contributions
                  + draft.potentially_distinct_elements + draft.redundancy_risks)

    unknown = set()
    for statement in statements:
        for group in BRACKET.findall(statement):
            for token in group.split(","):
                token = token.strip()
                if token and token not in known:
                    unknown.add(token)
    return sorted(unknown)


def compare_research_idea(idea_profile, paper_profiles, llm=None):
    """Compare l'idée avec les articles analysés. Retourne un objet Comparison."""
    if not paper_profiles:
        return Comparison(notes=["No analyzed papers available: comparison is impossible."])

    llm = llm or structured_llm(ComparisonDraft)
    draft = invoke_with_retries(llm, build_prompt(idea_profile, paper_profiles))

    # Le code retrouve les vraies preuves ; le LLM n'a fourni que des identifiants.
    evidence, unknown_evidence = resolve_evidence(draft.evidence_ids, paper_profiles)

    notes = []
    if unknown_evidence:
        notes.append(f"Ignored unknown evidence IDs cited by the model: {unknown_evidence}")
    if not evidence:
        notes.append("No verifiable evidence cited: treat these statements as unsupported.")

    unknown_citations = find_unknown_citations(draft, paper_profiles)
    if unknown_citations:
        notes.append(f"Statements cite unknown IDs: {unknown_citations}")

    if len(paper_profiles) < MIN_PAPERS_FOR_NOVELTY:
        notes.append(
            f"Only {len(paper_profiles)} paper(s) analyzed (fewer than {MIN_PAPERS_FOR_NOVELTY}): "
            "'potentially distinct' elements are weakly supported."
        )

    data = draft.model_dump()
    data["evidence_ids"] = [e.evidence_id for e in evidence]   # on ne garde que les valides
    return Comparison(**data, evidence=evidence, notes=notes)