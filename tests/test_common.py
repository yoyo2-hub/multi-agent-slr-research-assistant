from agents.common import format_papers, format_idea, resolve_evidence
from tests.sample_data import IDEA_PROFILE, PAPER_PROFILES


def test_resolve_evidence_trouve_les_vraies_preuves():
    found, unknown = resolve_evidence(["E001", "E003"], PAPER_PROFILES)
    assert [e.evidence_id for e in found] == ["E001", "E003"]
    assert found[1].paper_id == "P2"
    assert unknown == []


def test_resolve_evidence_ignore_inventes_et_doublons():
    found, unknown = resolve_evidence(["E001", "E999", "E001"], PAPER_PROFILES)
    assert [e.evidence_id for e in found] == ["E001"]
    assert unknown == ["E999"]


def test_format_papers_affiche_les_ids_de_preuves():
    text = format_papers(PAPER_PROFILES)
    assert "PAPER P1" in text
    assert "[E002]" in text


def test_format_papers_tronque_les_preuves_longues():
    profiles = [{"paper_id": "P9", "evidence": [
        {"evidence_id": "E900", "paper_id": "P9", "text": "x" * 1000}]}]
    assert "x" * 301 not in format_papers(profiles, max_evidence_chars=300)


def test_format_idea():
    assert "- keywords: time series anomaly detection; contrastive learning; graph attention" in format_idea(IDEA_PROFILE)