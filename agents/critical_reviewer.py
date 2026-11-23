import re

from agents.common import (bracket_ids, format_comparison, format_gaps, format_idea,
                           invoke_with_retries, unknown_bracket_ids)
from graph.schemas import CriticalReview, ReviewDraft, VerificationRequest
from llm import structured_llm

MIN_PAPERS_FOR_REVIEW = 5
TAG = "(reviewer judgment)"
COMPARISON_KEYS = ("similarities", "differences", "existing_contributions",
                   "potentially_distinct_elements", "redundancy_risks")
LIST_FIELDS = ("strengths", "weaknesses", "methodological_issues",
               "novelty_risks", "missing_evidence", "recommendations")


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
- Use ONLY the information above.
- Rely on the comparison and the gaps: when they show a limitation or an overlap, use it in your
  points and cite the paper, e.g. "[P1]". Points that ignore them are generic.
- Cite a paper only if the information above actually supports the statement.
- Any statement that mentions a paper must cite it in square brackets, e.g. "[P1]", never as
  bare text like P1. A statement that mentions no paper and rests only on your own reasoning
  must end with "(reviewer judgment)". Never put "(reviewer judgment)" on a statement that
  mentions a paper.
- A limitation of a paper that the idea addresses supports the idea being distinct; it is not
  a novelty risk. A novelty risk is something an analyzed paper already does close to the idea.
- Never say that the idea is proven novel, or that it is not novel. Novelty is only a risk here.
- Gaps with low confidence are weak support: say so instead of relying on them.
- "missing_evidence": claims or assumptions of the idea that nothing above supports. For each
  item, explain in weaknesses or methodological_issues why it matters.
- Do not repeat the same point in several lists.
- "verification_requests": only if a specific check could change your conclusion. At most 3.
  Use kind "search_more_literature" with a short search query whenever the check is a
  literature search. Use "check_evidence" only to verify a claim against the analyzed papers.
- "feasibility_level" is low, medium or high, with a justification in "feasibility".
- "feasibility_level" cannot be "high" if something required to carry out the idea (data,
  resources, a reliable component) appears in missing_evidence or verification_requests.
- At most 4 items per list. Leave a list empty rather than inventing items.
- Write with a space between every two words.
"""


def known_ids_from_input(comparison, gaps):
    """IDs the review is allowed to cite: those present in its inputs."""
    known = bracket_ids(format_comparison(comparison))
    for gap in gaps.get("research_gaps") or []:
        known.update(gap.get("supporting_papers", []))
        known.update(e["evidence_id"] for e in gap.get("evidence", []))
    return known


def bracket_bare_ids(statement, known_ids):
    """Wrap known IDs written as bare text, unless the statement already cites them.

    'P1 ignores X'     -> '[P1] ignores X'
    'limit of P1 [P1]' -> unchanged (already cited)
    Returns (statement, changed).
    """
    if not known_ids:
        return statement, False

    already = bracket_ids(statement)
    todo = [i for i in known_ids if i not in already]
    if not todo:
        return statement, False

    ids = sorted(todo, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(i) for i in ids) + r")\b")

    parts = re.split(r"(\[[^\]]*\])", statement)   # odd parts are inside brackets
    changed = False
    for k in range(0, len(parts), 2):
        new = pattern.sub(r"[\1]", parts[k])
        if new != parts[k]:
            parts[k] = new
            changed = True
    return "".join(parts), changed


def label_statement(statement):
    """Fix the label: a cited statement has no tag, an uncited one has the tag.

    Returns (statement, changed).
    """
    has_citation = bool(bracket_ids(statement))
    has_tag = TAG in statement
    if has_citation and has_tag:
        return statement.replace(TAG, "").rstrip(), True
    if not has_citation and not has_tag:
        return statement.rstrip() + " " + TAG, True
    return statement, False


def normalize_statements(data, known_ids):
    """Bracket bare IDs, then fix labels, on every statement of the review (in place).

    Returns (statements_bracketed, statements_relabeled).
    """
    bracketed = relabeled = 0

    def fix(statement):
        nonlocal bracketed, relabeled
        statement = re.sub(r"\s+([.,;:])", r"\1", statement)
        statement, b = bracket_bare_ids(statement, known_ids)
        statement, r = label_statement(statement)
        bracketed += b
        relabeled += r
        return statement

    for field in LIST_FIELDS:
        data[field] = [fix(s) for s in data[field]]
    data["feasibility"] = fix(data["feasibility"])
    return bracketed, relabeled


def review_research_idea(idea_profile, comparison, gaps, corpus_size=None, llm=None):
    """Critique the idea. corpus_size is optional and only used for the code-side checks."""
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

    known = known_ids_from_input(comparison, gaps)
    data = draft.model_dump()
    bracketed, relabeled = normalize_statements(data, known)

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

    statements = [s for field in LIST_FIELDS for s in data[field]] + [data["feasibility"]]
    unknown = unknown_bracket_ids(statements, known)
    if unknown:
        notes.append(f"Statements cite unknown IDs: {unknown}")

    if bracketed:
        notes.append(f"{bracketed} statement(s) mentioned a paper without brackets: fixed by the code.")
    if relabeled:
        notes.append(f"{relabeled} statement(s) had a missing or misplaced '{TAG}' label: "
                     "fixed by the code.")

    if draft.feasibility_level == "high" and draft.verification_requests:
        notes.append("Feasibility is 'high' but verification requests are open: treat it as provisional.")

    return CriticalReview(**data, needs_more_evidence=needs_more, notes=notes)