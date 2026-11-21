import re
import unicodedata


def normalize_doi(doi):
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def normalize_title(title):
    """'Deep Learning: A Survey!' et 'deep learning a survey' -> même clé."""
    if not title:
        return ""
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))  # retire les accents
    title = re.sub(r"[^a-z0-9\s]", " ", title.lower())                # retire la ponctuation
    return re.sub(r"\s+", " ", title).strip()


def merge_papers(kept, other):
    """Complète l'article gardé avec les champs manquants de l'autre."""
    for field in ("abstract", "doi", "year", "url"):
        if getattr(kept, field) is None and getattr(other, field) is not None:
            setattr(kept, field, getattr(other, field))
    if not kept.authors and other.authors:
        kept.authors = other.authors


def deduplicate(papers):
    """Retourne (articles_uniques, journal_des_doublons)."""
    unique = []
    by_doi = {}
    by_title = {}
    log = []

    for paper in papers:
        doi = normalize_doi(paper.doi)
        tkey = normalize_title(paper.title)

        existing, reason = None, None
        if doi and doi in by_doi:
            existing, reason = by_doi[doi], "doi"
        elif tkey and tkey in by_title:
            existing, reason = by_title[tkey], "titre"

        if existing:
            merge_papers(existing, paper)
            log.append({"kept": existing.paper_id, "removed": paper.paper_id, "reason": reason})
            # l'article gardé a peut-être gagné un DOI grâce à la fusion
            new_doi = normalize_doi(existing.doi)
            if new_doi:
                by_doi[new_doi] = existing
        else:
            unique.append(paper)
            if doi:
                by_doi[doi] = paper
            if tkey:
                by_title[tkey] = paper

    return unique, log