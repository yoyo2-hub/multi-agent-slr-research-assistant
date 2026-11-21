from graph.schemas import Paper
from slr.deduplication import deduplicate, normalize_doi, normalize_title


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"
    assert normalize_doi(None) is None


def test_normalize_title():
    assert normalize_title("Deep Learning: A Survey!") == normalize_title("deep  learning a survey")
    assert normalize_title("Détection d'anomalies") == "detection d anomalies"


def test_dedup_par_doi():
    a = Paper(paper_id="W1", title="Titre A", doi="10.1/abc")
    b = Paper(paper_id="W2", title="Titre different", doi="https://doi.org/10.1/ABC", abstract="Resume")
    unique, log = deduplicate([a, b])
    assert len(unique) == 1
    assert unique[0].abstract == "Resume"   # récupéré depuis le doublon
    assert log == [{"kept": "W1", "removed": "W2", "reason": "doi"}]


def test_dedup_par_titre_sans_doi():
    a = Paper(paper_id="W1", title="Anomaly Detection: A Survey")
    b = Paper(paper_id="W2", title="anomaly detection a survey")
    unique, log = deduplicate([a, b])
    assert len(unique) == 1
    assert log[0]["reason"] == "titre"


def test_articles_differents_restent():
    a = Paper(paper_id="W1", title="Article A", doi="10.1/a")
    b = Paper(paper_id="W2", title="Article B", doi="10.1/b")
    unique, log = deduplicate([a, b])
    assert len(unique) == 2
    assert log == []