import json
import os
import sqlite3

from graph.schemas import Paper

DB_PATH = "data/slr.sqlite"


def get_connection(path=DB_PATH):
    """Ouvre la base (la crée si besoin) avec ses tables."""
    if path != ":memory:":
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id         TEXT PRIMARY KEY,
            title            TEXT NOT NULL,
            abstract         TEXT,
            doi              TEXT,
            year             INTEGER,
            authors          TEXT,
            url              TEXT,
            pdf_path         TEXT,
            included         INTEGER NOT NULL DEFAULT 1,
            screening_reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS duplicates (
            kept_id    TEXT NOT NULL,
            removed_id TEXT NOT NULL,
            reason     TEXT NOT NULL,
            UNIQUE (kept_id, removed_id)
        )
    """)
    conn.commit()
    return conn


def save_papers(conn, papers):
    """Enregistre les articles. Un article déjà présent est conservé tel quel."""
    inserted = 0
    with conn:
        for p in papers:
            cur = conn.execute(
                "INSERT OR IGNORE INTO papers "
                "(paper_id, title, abstract, doi, year, authors, url, pdf_path, included, screening_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (p.paper_id, p.title, p.abstract, p.doi, p.year,
                 json.dumps(p.authors, ensure_ascii=False), p.url, p.pdf_path,
                 int(p.included), p.screening_reason),
            )
            inserted += cur.rowcount
    return inserted


def save_duplicates(conn, log):
    """Enregistre le journal des doublons fusionnés."""
    with conn:
        for entry in log:
            conn.execute(
                "INSERT OR IGNORE INTO duplicates (kept_id, removed_id, reason) VALUES (?, ?, ?)",
                (entry["kept"], entry["removed"], entry["reason"]),
            )


def update_screening(conn, paper_id, included, reason):
    """Enregistre la décision de screening d'un article."""
    with conn:
        conn.execute(
            "UPDATE papers SET included = ?, screening_reason = ? WHERE paper_id = ?",
            (int(included), reason, paper_id),
        )


def load_papers(conn, included_only=False, unscreened_only=False):
    """Relit les articles et reconstruit des objets Paper."""
    conditions = []
    if included_only:
        conditions.append("included = 1")
    if unscreened_only:
        conditions.append("screening_reason IS NULL")

    query = "SELECT * FROM papers"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    papers = []
    for row in conn.execute(query):
        papers.append(Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            abstract=row["abstract"],
            doi=row["doi"],
            year=row["year"],
            authors=json.loads(row["authors"] or "[]"),
            url=row["url"],
            pdf_path=row["pdf_path"],
            included=bool(row["included"]),
            screening_reason=row["screening_reason"],
        ))
    return papers