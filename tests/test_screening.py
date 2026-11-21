from graph.schemas import Paper
from slr.screening import ScreeningDecision, rule_based_screen, screen_papers

LONG = "x" * 150


class FakeLLM:
    def invoke(self, prompt):
        decision = "exclude" if "BAD" in prompt else "include"
        return ScreeningDecision(decision=decision, justification="test")


def test_regles():
    assert rule_based_screen(Paper(paper_id="1", title="a", abstract=None, year=2022), 2018)
    assert rule_based_screen(Paper(paper_id="2", title="a", abstract=LONG, year=2015), 2018)
    assert rule_based_screen(Paper(paper_id="3", title="a", abstract=LONG, year=None), 2018)
    assert rule_based_screen(Paper(paper_id="4", title="a", abstract=LONG, year=2022), 2018) is None


def test_screen_papers_avec_faux_llm():
    papers = [
        Paper(paper_id="1", title="Good", abstract=LONG, year=2022),
        Paper(paper_id="2", title="BAD", abstract=LONG, year=2022),
        Paper(paper_id="3", title="Old", abstract=LONG, year=2010),
    ]
    saved = []
    stats = screen_papers(papers, 2018, llm=FakeLLM(), delay=0, on_decision=saved.append)

    assert stats == {"included": 1, "excluded_by_llm": 1, "excluded_by_rules": 1, "failed": 0}
    assert papers[0].included is True
    assert papers[1].included is False
    assert papers[2].screening_reason.startswith("[rule]")
    assert len(saved) == 3