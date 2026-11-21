import time

from graph.schemas import Evidence


def format_idea(idea_profile):
    """Met le profil de l'idée en texte lisible pour le prompt."""
    lines = []
    for key, value in idea_profile.items():
        if isinstance(value, list):
            value = "; ".join(value)
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def format_papers(paper_profiles, max_papers=15, max_evidence_chars=300):
    """Résume les profils d'articles. Les limites protègent le quota de tokens."""
    blocks = []
    for p in paper_profiles[:max_papers]:
        lines = [
            f"PAPER {p['paper_id']}",
            f"  problem: {p.get('problem', '')}",
            f"  method: {p.get('method', '')}",
            f"  contributions: {'; '.join(p.get('contributions', []))}",
            f"  limitations: {'; '.join(p.get('limitations', []))}",
            f"  future_work: {'; '.join(p.get('future_work', []))}",
        ]
        for ev in p.get("evidence", []):
            lines.append(f"  [{ev['evidence_id']}] ({ev.get('section') or 'n/a'}) "
                         f"{ev['text'][:max_evidence_chars]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def resolve_evidence(evidence_ids, paper_profiles):
    """Retrouve les vraies preuves à partir de leurs identifiants.

    Retourne (preuves_valides, identifiants_inconnus).
    """
    index = {}
    for p in paper_profiles:
        for ev in p.get("evidence", []):
            index[ev["evidence_id"]] = Evidence(**ev)

    found, unknown, seen = [], [], set()
    for eid in evidence_ids:
        if eid in seen:
            continue
        seen.add(eid)
        if eid in index:
            found.append(index[eid])
        else:
            unknown.append(eid)
    return found, unknown


def invoke_with_retries(llm, prompt, retries=3):
    """Appelle le LLM avec quelques essais (réseau instable, limite de débit)."""
    last_error = None
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            print(f"  [essai {attempt + 1}/{retries}] {type(e).__name__}: {str(e)[:200]}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts") from last_error

def format_comparison(comparison):
    """Met le résultat de l'Agent 4 en texte lisible pour le prompt."""
    sections = ["similarities", "differences", "existing_contributions",
                "potentially_distinct_elements", "redundancy_risks"]
    lines = []
    for name in sections:
        items = comparison.get(name) or []
        if items:
            lines.append(f"{name}:")
            lines.extend(f"  - {item}" for item in items)
    return "\n".join(lines) or "(empty comparison)"