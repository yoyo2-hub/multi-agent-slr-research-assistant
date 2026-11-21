from agents.common import (format_comparison, format_papers, invoke_with_retries,
                           resolve_evidence)
from graph.schemas import GapsDraft, ResearchGap, ResearchGaps
from llm import structured_llm

MIN_PAPERS_FOR_HIGH = 5
RANK = {"high": 3, "medium": 2, "low": 1}


def compute_confidence(n_supporting_papers, corpus_size):
    """Confiance déduite du nombre d'articles distincts qui appuient la lacune."""
    if n_supporting_papers >= 3:
        level = "high"
    elif n_supporting_papers == 2:
        level = "medium"
    else:
        level = "low"
    if corpus_size < MIN_PAPERS_FOR_HIGH and level == "high":
        level = "medium"
    return level


def build_prompt(comparison, paper_profiles):
    return f"""You are a rigorous research analyst. Identify candidate research gaps in the literature.

COMPARISON OF THE RESEARCH IDEA WITH THE LITERATURE:
{format_comparison(comparison)}

ANALYZED PAPERS (evidence passages are in brackets, e.g. [E001]):
{format_papers(paper_profiles)}

A research gap is something the analyzed papers leave open: a stated limitation, a stated
future-work direction, or a clear lack of coverage.

Gap types: methodological, technical, empirical, evaluation, integration, theoretical.

Rules:
- Use ONLY the information above. Do not rely on outside knowledge.
- Propose at most 5 gaps. Fewer is better than weak ones.
- Every gap MUST cite at least one evidence ID in evidence_ids, taken from the passages above
  (preferably limitations or future work). Never invent an ID.
- Do not propose a gap for which you cannot cite evidence: return fewer gaps, or an empty list.
- Scope each gap to the cited paper(s). Write "P1 does not model ..." or "The benchmark of P2
  contains few ...", never claims about a whole field such as "existing methods lack",
  "benchmarks lack" or "no work addresses".
- Write descriptions with a space between every two words.
- Do not claim that a gap is proven or that nobody has addressed it. These are candidates only.
- One idea per gap; do not repeat the same gap in different words.
- Write each description in one or two clear sentences.
"""


def discover_research_gaps(comparison, paper_profiles, llm=None):
    """Identifie des lacunes candidates, avec preuves et confiance calculées par le code."""
    if not paper_profiles:
        return ResearchGaps(notes=["No analyzed papers available: gap discovery is impossible."])

    llm = llm or structured_llm(GapsDraft)
    draft = invoke_with_retries(llm, build_prompt(comparison, paper_profiles))

    corpus_size = len(paper_profiles)
    gaps, notes = [], []

    for gap in draft.research_gaps:
        evidence, unknown = resolve_evidence(gap.evidence_ids, paper_profiles)
        if unknown:
            notes.append(f"Ignored unknown evidence IDs {unknown} for gap: '{gap.description[:60]}'")
        if not evidence:
            notes.append(f"Dropped gap without verifiable evidence: '{gap.description[:80]}'")
            continue

        supporting = sorted({e.paper_id for e in evidence})
        gaps.append(ResearchGap(
            description=gap.description,
            type=gap.type,
            supporting_papers=supporting,
            evidence=evidence,
            confidence=compute_confidence(len(supporting), corpus_size),
        ))

    gaps.sort(key=lambda g: (RANK[g.confidence], len(g.supporting_papers)), reverse=True)

    if not gaps:
        notes.append("No evidence-backed gap found.")
    if corpus_size < MIN_PAPERS_FOR_HIGH:
        notes.append(f"Only {corpus_size} paper(s) analyzed: confidence cannot exceed 'medium'.")

    return ResearchGaps(research_gaps=gaps, notes=notes)