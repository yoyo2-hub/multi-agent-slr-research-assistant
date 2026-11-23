import copy

from agents.critical_reviewer import label_statement, review_research_idea
from graph.schemas import ReviewDraft, VerificationRequest
from tests.sample_data import COMPARISON, GAPS, IDEA_PROFILE


class FakeLLM:
    def __init__(self, draft):
        self.draft = draft
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return self.draft


def draft(**kwargs):
    base = dict(feasibility_level="medium", feasibility="Reasonable scope. (reviewer judgment)")
    base.update(kwargs)
    return ReviewDraft(**base)


def strong_gaps():
    gaps = copy.deepcopy(GAPS)
    gaps["research_gaps"][0]["confidence"] = "medium"
    return gaps


def test_demande_de_verification_declenche_needs_more_evidence():
    d = draft(verification_requests=[VerificationRequest(
        kind="search_more_literature", reason="Few papers", query="graph attention anomaly detection")])
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=10, llm=FakeLLM(d))
    assert result.needs_more_evidence is True
    assert result.verification_requests[0].query == "graph attention anomaly detection"


def test_aucun_declencheur_pas_de_boucle():
    d = draft(novelty_risks=["Overlap with contrastive methods [P1]"])
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=10, llm=FakeLLM(d))
    assert result.needs_more_evidence is False
    assert result.notes == []


def test_petit_corpus_declenche_une_note():
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=2, llm=FakeLLM(draft()))
    assert result.needs_more_evidence is True
    assert any("weakly grounded" in n for n in result.notes)


def test_lacunes_faibles_declenchent_une_note():
    result = review_research_idea(IDEA_PROFILE, COMPARISON, GAPS, corpus_size=10, llm=FakeLLM(draft()))
    assert result.needs_more_evidence is True
    assert any("No gap with medium or high confidence" in n for n in result.notes)


def test_redondance_signalee_mais_non_reprise():
    comparison = copy.deepcopy(COMPARISON)
    comparison["redundancy_risks"] = ["Overlap with P1 [P1]"]
    result = review_research_idea(IDEA_PROFILE, comparison, strong_gaps(), corpus_size=10,
                                  llm=FakeLLM(draft(novelty_risks=[])))
    assert any("redundancy risks" in n for n in result.notes)


def test_citation_inconnue_signalee():
    d = draft(weaknesses=["Weak baseline [P1, P9]"])
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=10, llm=FakeLLM(d))
    note = next(n for n in result.notes if "unknown IDs" in n)
    assert "P9" in note and "P1" not in note


def test_rien_a_critiquer_demande_une_recherche_sans_appeler_le_llm():
    llm = FakeLLM(draft())
    result = review_research_idea(IDEA_PROFILE, {}, {}, llm=llm)
    assert llm.last_prompt is None
    assert result.needs_more_evidence is True
    request = result.verification_requests[0]
    assert request.kind == "search_more_literature"
    assert "time series anomaly detection" in request.query


def test_idee_vide_pas_d_appel_llm():
    llm = FakeLLM(draft())
    result = review_research_idea({}, COMPARISON, GAPS, llm=llm)
    assert llm.last_prompt is None
    assert "Not assessed" in result.feasibility


def test_prompt_contient_entrees_et_regles():
    llm = FakeLLM(draft())
    review_research_idea(IDEA_PROFILE, COMPARISON, GAPS, corpus_size=2, llm=llm)
    assert "Graph attention over sensors" in llm.last_prompt
    assert "confidence: low" in llm.last_prompt
    assert "Critique the research idea" in llm.last_prompt
    assert "(reviewer judgment)" in llm.last_prompt
    assert "2 analyzed paper(s)" in llm.last_prompt
    assert "never as" in llm.last_prompt and "bare text" in llm.last_prompt
    assert "Rely on the comparison and the gaps" in llm.last_prompt
    assert "why it matters" in llm.last_prompt

def test_mention_nue_d_un_article_signalee():
    d = draft(
        strengths=["Addresses a limit of the benchmark P2. (reviewer judgment)"],
        weaknesses=["Fine point [P1] (reviewer judgment)", "Pure opinion. (reviewer judgment)"],
    )
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=10, llm=FakeLLM(d))
    note = next(n for n in result.notes if "without a bracket citation" in n)
    assert "1 statement(s)" in note
    assert "P2" in note


def test_faisabilite_elevee_avec_verifications_ouvertes():
    d = draft(feasibility_level="high", verification_requests=[
        VerificationRequest(kind="check_evidence", reason="Dataset availability")])
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=10, llm=FakeLLM(d))
    assert any("provisional" in n for n in result.notes)

def test_label_statement():
    assert label_statement("Opinion") == ("Opinion (reviewer judgment)", True)
    assert label_statement("Fact [P1]") == ("Fact [P1]", False)
    assert label_statement("Fact [P1] (reviewer judgment)") == ("Fact [P1]", True)
    assert label_statement("Opinion (reviewer judgment)") == ("Opinion (reviewer judgment)", False)


def test_etiquettes_corrigees_par_le_code():
    d = draft(
        strengths=["Plain opinion"],
        weaknesses=["Based on a paper [P1] (reviewer judgment)"],
        recommendations=["Already tagged (reviewer judgment)"],
    )
    result = review_research_idea(IDEA_PROFILE, COMPARISON, strong_gaps(), corpus_size=10, llm=FakeLLM(d))
    assert result.strengths == ["Plain opinion (reviewer judgment)"]
    assert result.weaknesses == ["Based on a paper [P1]"]
    assert result.recommendations == ["Already tagged (reviewer judgment)"]
    assert any("fixed by the code" in n for n in result.notes)