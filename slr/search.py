import config

import requests

from graph.schemas import Paper


def reconstruct_abstract(inverted_index):
    # Cas vide
    if not inverted_index:
        return None

    # Liste de couples (position, mot)
    couples = []
    for mot, positions in inverted_index.items():
        for pos in positions:
            couples.append((pos, mot))

    # Tri par position, puis assemblage des mots
    couples.sort()
    return " ".join(mot for pos, mot in couples)


def parse_work(work):
    # Un article sans titre est inutilisable
    title = work.get("title")
    if not title:
        return None

    # "https://openalex.org/W123" -> "W123"
    paper_id = (work.get("id") or "").split("/")[-1]
    if not paper_id:
        return None

    # "https://doi.org/10.1000/xyz" -> "10.1000/xyz"
    doi = work.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "")

    # Liste des noms d'auteurs
    authors = []
    for a in work.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append(name)

    # URL : page de l'article, sinon le lien DOI
    location = work.get("primary_location") or {}
    url = location.get("landing_page_url")
    if not url and doi:
        url = "https://doi.org/" + doi

    return Paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        doi=doi,
        year=work.get("publication_year"),
        url=url,
    )

OPENALEX_URL = "https://api.openalex.org/works"

def search_openalex(query, year_min=None, max_results=50):
    """Cherche des articles pour UNE requête. Retourne une liste de Paper."""
    year_min = year_min or config.YEAR_MIN
    papers = []
    cursor = "*"  # "*" = première page ; ensuite OpenAlex fournit le curseur suivant

    while len(papers) < max_results:
        params = {
            "search": query,
            "filter": f"from_publication_date:{year_min}-01-01,type:article,has_abstract:true",
            "per_page": min(100, max_results - len(papers)),  # 100 = maximum autorisé
            "cursor": cursor,
        }
        if config.OPENALEX_API_KEY:
            params["api_key"] = config.OPENALEX_API_KEY
        if config.OPENALEX_EMAIL:
            params["mailto"] = config.OPENALEX_EMAIL

        response = requests.get(OPENALEX_URL, params=params, timeout=30)
        response.raise_for_status()  # lève une erreur si 401, 403, 429, 500...
        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            paper = parse_work(work)
            if paper:  # on ignore les articles inutilisables (None)
                papers.append(paper)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:  # plus de pages
            break

    return papers[:max_results]


def search_papers(queries, year_min=None, max_per_query=30):
    """Lance plusieurs requêtes et regroupe tous les Paper (sans dédupliquer)."""
    all_papers = []
    for q in queries:
        all_papers.extend(search_openalex(q, year_min, max_per_query))
    return all_papers