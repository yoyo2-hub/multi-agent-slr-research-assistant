from agents.comparison_agent import compare_research_idea
from graph.schemas import ComparisonDraft
from tests.sample_data import IDEA_PROFILE, PAPER_PROFILES


class FakeLLM:
    def __init__(self, draft):
        self.draft = draft
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return self.draft


def test_preuves_resolues_et_ids_inventes_ignores():
    draft = ComparisonDraft(
        similarities=["Both use contrastive learning [P1]"],
        evidence_ids=["E001", "E999", "E001"],   # E999 est inventé, E001 en double
    )
    result = compare_research_idea(IDEA_PROFILE, PAPER_PROFILES, llm=FakeLLM(draft))

    assert result.evidence_ids == ["E001"]
    assert [e.evidence_id for e in result.evidence] == ["E001"]
    assert result.evidence[0].paper_id == "P1"
    assert "contrasting augmented views" in result.evidence[0].text   # vrai texte des profils
    assert any("E999" in note for note in result.notes)


def test_aucune_preuve_citee_ajoute_une_note():
    draft = ComparisonDraft(similarities=["Something [P1]"], evidence_ids=[])
    result = compare_research_idea(IDEA_PROFILE, PAPER_PROFILES, llm=FakeLLM(draft))
    assert result.evidence == []
    assert any("unsupported" in note for note in result.notes)


def test_sans_articles_pas_d_appel_llm():
    llm = FakeLLM(ComparisonDraft())
    result = compare_research_idea(IDEA_PROFILE, [], llm=llm)
    assert llm.last_prompt is None
    assert result.evidence == []
    assert result.notes


def test_prompt_contient_idee_preuves_et_regles():
    llm = FakeLLM(ComparisonDraft(evidence_ids=["E001"]))
    compare_research_idea(IDEA_PROFILE, PAPER_PROFILES, llm=llm)
    assert "graph attention" in llm.last_prompt
    assert "[E002]" in llm.last_prompt
    assert "NOT proven novelty" in llm.last_prompt

def test_citation_inconnue_signalee():
    draft = ComparisonDraft(similarities=["Uses X [P1, P9]"], evidence_ids=["E001"])
    result = compare_research_idea(IDEA_PROFILE, PAPER_PROFILES, llm=FakeLLM(draft))
    note = next(n for n in result.notes if "cite unknown" in n)
    assert "P9" in note
    assert "P1" not in note


def test_petit_corpus_signale():
    draft = ComparisonDraft(evidence_ids=["E001"])
    result = compare_research_idea(IDEA_PROFILE, PAPER_PROFILES, llm=FakeLLM(draft))
    assert any("weakly supported" in n for n in result.notes)


def test_prompt_contient_les_regles_anti_invention():
    llm = FakeLLM(ComparisonDraft(evidence_ids=["E001"]))
    compare_research_idea(IDEA_PROFILE, PAPER_PROFILES, llm=llm)
    assert "not proof of absence" in llm.last_prompt
    assert "A paper's limitation is not a redundancy risk" in llm.last_prompt