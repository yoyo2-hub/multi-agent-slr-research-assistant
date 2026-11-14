"""
rag/ingestion.py - Etape 1 du RAG : lire les articles.

Pour chaque article de data/papers.json :
  - si le PDF existe  -> on lit le texte page par page (on garde le numero de page)
  - sinon             -> on utilise le resume (abstract) comme texte de secours
"""
import json
from pathlib import Path

import pymupdf  # PyMuPDF


def extract_pdf(pdf_path):
    """Lit un PDF et renvoie une liste de pages : [{'page': 1, 'text': '...'}, ...]"""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": number, "text": text})
    return pages


def ingest_all(papers_json="data/papers.json"):
    """Renvoie la liste de tous les textes a decouper, avec leurs metadonnees."""
    papers = json.loads(Path(papers_json).read_text(encoding="utf-8"))
    documents = []

    for paper in papers:
        if not paper.get("included", True):
            continue

        meta = {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "doi": paper.get("doi"),
            "year": paper.get("year"),
        }

        pdf_path = paper.get("pdf_path")
        pages = []
        if pdf_path and Path(pdf_path).exists():
            try:
                pages = extract_pdf(pdf_path)
            except Exception as exc:
                print(f"  ! PDF illisible ({paper['paper_id']}) : {exc}")

        if pages:
            for p in pages:
                documents.append({**meta, "page": p["page"], "section": "fulltext", "text": p["text"]})
        elif paper.get("abstract"):  # repli : pas de PDF -> on garde le resume
            documents.append({**meta, "page": 0, "section": "abstract", "text": paper["abstract"]})

    return documents


if __name__ == "__main__":
    docs = ingest_all()
    print(f"{len(docs)} pages / textes lus")