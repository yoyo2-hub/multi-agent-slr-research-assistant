from agents.gap_discovery_agent import compute_confidence, discover_research_gaps
from graph.schemas import GapDraft, GapsDraft
from tests.sample_data import COMPARISON, PAPER_PROFILES


class FakeLLM:
    def __init__(self, draft):
        self.draft = draft
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return self.draft


def test_compute_confidence():
    assert compute_confidence(0, 10) == "low"
    assert compute_confidence(1, 10) == "low"
    assert compute_confidence(2, 10) == "medium"
    assert compute_confidence(3, 10) == "high"
    assert compute_confidence(3, 2) == "medium"     # corpus trop petit : plafonné


def test_lacune_avec_deux_articles_est_medium_et_triee_en_premier():
    draft = GapsDraft(research_gaps=[
        GapDraft(description="Few labels", type="empirical", evidence_ids=["E001"]),
        GapDraft(description="Sensor dependencies and industrial data", type="technical",
                 evidence_ids=["E002", "E003"]),
    ])
    result = discover_research_gaps(COMPARISON, PAPER_PROFILES, llm=FakeLLM(draft))

    assert [g.confidence for g in result.research_gaps] == ["medium", "low"]
    best = result.research_gaps[0]
    assert best.supporting_papers == ["P1", "P2"]
    assert [e.evidence_id for e in best.evidence] == ["E002", "E003"]
    assert "does not model inter-sensor relations" in best.evidence[0].text


def test_lacune_sans_preuve_valide_est_ecartee():
    draft = GapsDraft(research_gaps=[
        GapDraft(description="Invented gap", type="theoretical", evidence_ids=["E999"]),
        GapDraft(description="Real gap", type="empirical", evidence_ids=["E003"]),
    ])
    result = discover_research_gaps(COMPARISON, PAPER_PROFILES, llm=FakeLLM(draft))

    assert [g.description for g in result.research_gaps] == ["Real gap"]
    assert any("Dropped gap" in n for n in result.notes)
    assert any("E999" in n for n in result.notes)


def test_id_inconnu_melange_a_un_id_valide():
    draft = GapsDraft(research_gaps=[
        GapDraft(description="Partly invented", type="empirical", evidence_ids=["E001", "E999"]),
    ])
    result = discover_research_gaps(COMPARISON, PAPER_PROFILES, llm=FakeLLM(draft))

    assert len(result.research_gaps) == 1
    assert [e.evidence_id for e in result.research_gaps[0].evidence] == ["E001"]
    assert any("E999" in n for n in result.notes)


def test_sans_articles_pas_d_appel_llm():
    llm = FakeLLM(GapsDraft())
    result = discover_research_gaps(COMPARISON, [], llm=llm)
    assert llm.last_prompt is None
    assert result.research_gaps == [] and result.notes


def test_prompt_contient_comparaison_et_regles():
    llm = FakeLLM(GapsDraft())
    discover_research_gaps(COMPARISON, PAPER_PROFILES, llm=llm)
    assert "Graph attention over sensors" in llm.last_prompt
    assert "[E002]" in llm.last_prompt
    assert "MUST cite at least one evidence ID" in llm.last_prompt