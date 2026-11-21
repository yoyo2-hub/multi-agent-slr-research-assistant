from database.db import get_connection, save_papers, save_duplicates, load_papers
from graph.schemas import Paper


def test_sauvegarde_et_relecture():
    conn = get_connection(":memory:")
    p = Paper(paper_id="W1", title="Titre", authors=["Alice", "Bob"],
              abstract="Resume", doi="10.1/a", year=2022)
    assert save_papers(conn, [p]) == 1

    relus = load_papers(conn)
    assert len(relus) == 1
    assert relus[0] == p   # même contenu qu'à l'aller


def test_pas_de_doublon_a_la_reinsertion():
    conn = get_connection(":memory:")
    p = Paper(paper_id="W1", title="Titre")
    save_papers(conn, [p])
    assert save_papers(conn, [p]) == 0
    assert len(load_papers(conn)) == 1


def test_included_only():
    conn = get_connection(":memory:")
    a = Paper(paper_id="W1", title="A")
    b = Paper(paper_id="W2", title="B", included=False)
    save_papers(conn, [a, b])
    assert len(load_papers(conn)) == 2
    assert [p.paper_id for p in load_papers(conn, included_only=True)] == ["W1"]


def test_journal_des_doublons():
    conn = get_connection(":memory:")
    log = [{"kept": "W1", "removed": "W2", "reason": "doi"}]
    save_duplicates(conn, log)
    save_duplicates(conn, log)  # relance : pas de double ligne
    n = conn.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0]
    assert n == 1