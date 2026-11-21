from slr.search import parse_work, reconstruct_abstract, search_openalex


def test_reconstruct_abstract():
    assert reconstruct_abstract({"World": [1], "Hello": [0], "again": [2]}) == "Hello World again"
    assert reconstruct_abstract({"the": [0, 3], "cat": [1], "sat": [2]}) == "the cat sat the"
    assert reconstruct_abstract(None) is None


def test_parse_work_complet():
    work = {
        "id": "https://openalex.org/W123",
        "title": "Anomaly detection in time series",
        "doi": "https://doi.org/10.1000/xyz",
        "publication_year": 2022,
        "authorships": [{"author": {"display_name": "Alice Martin"}}],
        "primary_location": {"landing_page_url": "https://example.org/paper"},
        "abstract_inverted_index": {"Hello": [0], "World": [1]},
    }
    p = parse_work(work)
    assert p.paper_id == "W123"
    assert p.doi == "10.1000/xyz"
    assert p.authors == ["Alice Martin"]
    assert p.abstract == "Hello World"
    assert p.year == 2022
    assert p.url == "https://example.org/paper"


def test_parse_work_champs_manquants():
    p = parse_work({"id": "https://openalex.org/W9", "title": "Minimal"})
    assert p.doi is None
    assert p.authors == []
    assert p.abstract is None
    assert p.url is None


def test_parse_work_sans_titre():
    assert parse_work({"id": "https://openalex.org/W1", "title": None}) is None

def test_search_openalex_mock(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"id": "https://openalex.org/W1", "title": "Article A"},
                    {"id": "https://openalex.org/W2", "title": None},
                ],
                "meta": {"next_cursor": None},
            }

    monkeypatch.setattr("slr.search.requests.get", lambda *a, **k: FakeResponse())
    papers = search_openalex("test", 2018, 10)
    assert len(papers) == 1
    assert papers[0].paper_id == "W1"