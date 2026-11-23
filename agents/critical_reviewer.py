from agents.common import (bracket_ids, format_comparison, format_gaps, format_idea,
                           invoke_with_retries, unknown_bracket_ids)
from graph.schemas import CriticalReview, ReviewDraft, VerificationRequest
from llm import structured_llm

MIN_PAPERS_FOR_REVIEW = 5
COMPARISON_KEYS = ("similarities", "differences", "existing_contributions",
                   "potentially_distinct_elements", "redundancy_risks")


def build_prompt(idea_profile, comparison, gaps, corpus_size):
    size = corpus_size if corpus_size is not None else "unknown"
    return f"""You are a critical reviewer evaluating a research idea, like a demanding conference reviewer.

RESEARCH IDEA:
{format_idea(idea_profile)}

COMPARISON WITH THE LITERATURE (from a previous analysis):
{format_comparison(comparison)}

CANDIDATE RESEARCH GAPS (confidence was computed from verifiable evidence):
{format_gaps(gaps)}

CORPUS SIZE: {size} analyzed paper(s)

Rules:
- Critique the research idea, not the literature.
- Use ONLY the information above. A statement based on the literature must cite the paper_id(s)
  in square brackets, e.g. "[P1]". A statement based only on your own reasoning about the idea
  must end with "(reviewer judgment)".
- Never say that the idea is proven novel, or that it is not novel. Novelty is only a risk here.
- Gaps with low confidence are weak support: say so instead of relying on them.
- "missing_evidence": claims or assumptions of the idea that nothing above supports.
- "verification_requests": only if a specific check could change your conclusion. At most 3.
  Use kind "search_more_literature" with a short search query, or "check_evidence".
- "feasibility_level" is low, medium or high, with a justification in "feasibility".
- At most 4 items per list. Leave a list empty rather than inventing items.
- Write with a space between every two words.
"""


def known_ids_from_input(comparison, gaps):
    """Identifiants que la critique a le droit de citer : ceux présents dans ses entrées."""
    known = bracket_ids(format_comparison(comparison))
    for gap in gaps.get("research_gaps") or []:
        known.update(gap.get("supporting_papers", []))
        known.update(e["evidence_id"] for e in gap.get("evidence", []))
    return known


def review_research_idea(idea_profile, comparison, gaps, corpus_size=None, llm=None):
    """Critique l'idée. corpus_size est facultatif et sert aux contrôles du code."""
    comparison = comparison or {}
    gaps = gaps or {}

    if not idea_profile:
        return CriticalReview(
            feasibility_level="low",
            feasibility="Not assessed: no idea profile available.",
            notes=["Review skipped: the research idea is empty."],
        )

    has_comparison = any(comparison.get(k) for k in COMPARISON_KEYS)
    has_gaps = bool(gaps.get("research_gaps"))
    if not has_comparison and not has_gaps:
        query = " ".join((idea_profile.get("keywords") or [])[:3]) or None
        return CriticalReview(
            feasibility_level="low",
            feasibility="Not assessed: no literature comparison or gap available.",
            missing_evidence=["No literature comparison or evidence-backed gap is available."],
            verification_requests=[VerificationRequest(
                kind="search_more_literature",
                reason="Nothing to review the idea against.",
                query=query,
            )],
            needs_more_evidence=True,
            notes=["Review skipped: nothing to review the idea against."],
        )

    llm = llm or structured_llm(ReviewDraft)
    draft = invoke_with_retries(llm, build_prompt(idea_profile, comparison, gaps, corpus_size))

    notes = []
    needs_more = bool(draft.verification_requests)
    if draft.verification_requests:
        notes.append(f"The reviewer requested {len(draft.verification_requests)} verification(s).")

    if corpus_size is not None and corpus_size < MIN_PAPERS_FOR_REVIEW:
        needs_more = True
        notes.append(f"Only {corpus_size} paper(s) analyzed (fewer than {MIN_PAPERS_FOR_REVIEW}): "
                     "the review is weakly grounded.")

    gap_list = gaps.get("research_gaps") or []
    if not any(g.get("confidence") in ("medium", "high") for g in gap_list):
        needs_more = True
        notes.append("No gap with medium or high confidence: gap-based conclusions are weak.")

    if comparison.get("redundancy_risks") and not draft.novelty_risks:
        notes.append("The comparison flagged redundancy risks but the review lists no novelty risk.")

    statements = (draft.strengths + draft.weaknesses + draft.methodological_issues
                  + draft.novelty_risks + draft.missing_evidence + draft.recommendations
                  + [draft.feasibility])
    unknown = unknown_bracket_ids(statements, known_ids_from_input(comparison, gaps))
    if unknown:
        notes.append(f"Statements cite unknown IDs: {unknown}")

    return CriticalReview(**draft.model_dump(), needs_more_evidence=needs_more, notes=notes)