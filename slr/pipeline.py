import config
from database.db import (get_connection, save_papers, save_duplicates,
                         load_papers, update_screening)
from slr.screening import screen_papers
from slr.search import search_papers
from slr.deduplication import deduplicate
from database.db import get_connection, save_papers, save_duplicates, load_papers


def run_collection(queries, year_min=None, max_per_query=30):
    """Recherche -> déduplication -> stockage. Retourne un petit bilan."""
    found = search_papers(queries, year_min, max_per_query)
    unique, log = deduplicate(found)

    conn = get_connection()
    new = save_papers(conn, unique)
    save_duplicates(conn, log)
    total = len(load_papers(conn))
    conn.close()

    return {"trouves": len(found), "uniques": len(unique),
            "doublons": len(log), "nouveaux": new, "total_en_base": total}

def run_screening(year_min=None, limit=None, delay=1.0):
    """Screene les articles pas encore décidés."""
    year_min = year_min or config.YEAR_MIN
    conn = get_connection()
    todo = load_papers(conn, unscreened_only=True)
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} articles à screener")

    def save(paper):
        update_screening(conn, paper.paper_id, paper.included, paper.screening_reason)

    try:
        return screen_papers(todo, year_min, delay=delay, on_decision=save)
    except KeyboardInterrupt:
        print("Interrompu : les décisions déjà prises sont sauvegardées.")
        return None
    finally:
        conn.close()