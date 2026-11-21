import time
from typing import Literal

from pydantic import BaseModel, Field

from llm import structured_llm
from slr.criteria import TOPIC, INCLUSION, EXCLUSION

MIN_ABSTRACT_CHARS = 100


class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude"]
    justification: str = Field(
        description="One or two sentences explaining the decision, citing the criteria."
    )


def rule_based_screen(paper, year_min):
    """Retourne un motif d'exclusion, ou None si l'article passe les règles."""
    if not paper.abstract or len(paper.abstract) < MIN_ABSTRACT_CHARS:
        return "Missing or too short abstract"
    if paper.year is None:
        return "Missing publication year"
    if paper.year < year_min:
        return f"Published before {year_min}"
    return None


def build_prompt(paper):
    inclusion = "\n".join(f"- {c}" for c in INCLUSION)
    exclusion = "\n".join(f"- {c}" for c in EXCLUSION)
    return f"""You are screening papers for a systematic literature review on: {TOPIC}.

Inclusion criteria:
{inclusion}

Exclusion criteria:
{exclusion}

Decide "include" if the paper meets the inclusion criteria and no exclusion criterion applies.
Decide "exclude" if an exclusion criterion applies or the inclusion criteria are clearly not met.
If you are genuinely unsure from the abstract alone, decide "include" (a human will check later).
Base your decision ONLY on the title and abstract below. Do not invent details.

Title: {paper.title}
Year: {paper.year}
Abstract: {paper.abstract}
"""


def screen_paper_llm(paper, llm, retries=3):
    """Appelle le LLM avec quelques essais (erreurs réseau, limite de débit)."""
    prompt = build_prompt(paper)
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            print(f"  [essai {attempt + 1}/{retries}] {paper.paper_id}: {type(e).__name__}")
            time.sleep(2 ** attempt)
    return None


def screen_papers(papers, year_min, llm=None, delay=1.0, on_decision=None):
    """Applique règles puis LLM. Modifie les Paper sur place et retourne un bilan."""
    stats = {"included": 0, "excluded_by_rules": 0, "excluded_by_llm": 0, "failed": 0}
    total = len(papers)

    for i, paper in enumerate(papers, 1):
        reason = rule_based_screen(paper, year_min)
        if reason:
            paper.included = False
            paper.screening_reason = "[rule] " + reason
            stats["excluded_by_rules"] += 1
        else:
            if llm is None:
                llm = structured_llm(ScreeningDecision)
            decision = screen_paper_llm(paper, llm)
            if decision is None:
                stats["failed"] += 1
                print(f"[{i}/{total}] ECHEC (reprise au prochain lancement) | {paper.title[:60]}")
                continue
            paper.included = decision.decision == "include"
            paper.screening_reason = "[llm] " + decision.justification
            stats["included" if paper.included else "excluded_by_llm"] += 1
            time.sleep(delay)

        if on_decision:
            on_decision(paper)
        print(f"[{i}/{total}] {'INCLUS' if paper.included else 'EXCLU '} | {paper.title[:60]}")

    return stats