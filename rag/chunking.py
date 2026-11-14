"""
rag/chunking.py - Etape 2 du RAG : decouper les textes en petits morceaux (chunks).

Regle simple : des morceaux d'environ 200 mots, avec 40 mots de chevauchement
pour ne pas couper une idee en deux. Chaque chunk garde sa source (article, page).
"""
import json
from pathlib import Path

from rag.ingestion import ingest_all

CHUNK_WORDS = 200   # taille d'un morceau
OVERLAP_WORDS = 40  # mots repetes entre deux morceaux consecutifs
MIN_WORDS = 30      # on ignore les morceaux trop courts (numeros de page, etc.)


def split_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = size - overlap
    pieces = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= MIN_WORDS or (start == 0 and piece):
            pieces.append(" ".join(piece))
        if start + size >= len(words):
            break
    return pieces


def make_chunks(documents):
    chunks = []
    counters = {}  # numero du chunk, par article
    for doc in documents:
        for piece in split_words(doc["text"]):
            n = counters.get(doc["paper_id"], 0) + 1
            counters[doc["paper_id"]] = n
            chunks.append(
                {
                    "chunk_id": f"{doc['paper_id']}::c{n:04d}",
                    "paper_id": doc["paper_id"],
                    "title": doc["title"],
                    "doi": doc["doi"],
                    "year": doc["year"],
                    "section": doc["section"],
                    "page": doc["page"],
                    "text": piece,
                }
            )
    return chunks


if __name__ == "__main__":
    docs = ingest_all()
    chunks = make_chunks(docs)
    Path("data/chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(docs)} pages lues -> {len(chunks)} chunks sauvegardes dans data/chunks.json")
    if chunks:
        c = chunks[0]
        print(f"\nExemple : {c['chunk_id']} (page {c['page']})\n{c['text'][:300]}...")