"""
rag/retriever.py - Etape 5 du RAG : la recherche HYBRIDE.

Deux facons de chercher, qui se completent :
  1. Recherche par SENS  (vecteurs, Qdrant)  -> trouve "attaque" quand on cherche "intrusion"
  2. Recherche par MOTS  (BM25)              -> trouve les termes exacts : sigles, noms de modeles ("HyDE", "CVE-2021")

On fusionne les deux classements avec la methode RRF (Reciprocal Rank Fusion) :
un chunk bien classe dans les DEUX listes passe en tete. Les doublons sont fusionnes.
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.indexing import get_client, vector_search

RRF_K = 60  # constante classique de la methode RRF

# Mots tres courants, sans interet pour une recherche par mots (BM25 les ignore)
STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "and", "or", "with", "by", "is", "are",
    "was", "were", "be", "as", "at", "from", "that", "this", "it", "its", "what", "which",
    "how", "we", "can", "has", "have", "not", "do", "does",
}


def tokenize(text):
    """Decoupe un texte en mots minuscules (utilise par BM25)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


class HybridRetriever:
    def __init__(self, chunks_json="data/chunks.json"):
        self.chunks = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.client = get_client()

    def close(self):
        self.client.close()

    @staticmethod
    def _passes(chunk, year_min, paper_id, section):
        if year_min is not None and (chunk.get("year") is None or chunk["year"] < year_min):
            return False
        if paper_id is not None and chunk["paper_id"] != paper_id:
            return False
        if section is not None and chunk["section"] != section:
            return False
        return True

    def bm25_search(self, question, top_k, year_min=None, paper_id=None, section=None):
        """Recherche par mots exacts : renvoie les chunks tries du meilleur au moins bon."""
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if scores[i] <= 0:
                break  # plus aucun mot en commun avec la question
            chunk = self.chunks[i]
            if self._passes(chunk, year_min, paper_id, section):
                results.append(chunk)
                if len(results) == top_k:
                    break
        return results

    def search(self, question, top_k=5, candidates=20, year_min=None, paper_id=None, section=None):
        """Recherche hybride. Renvoie les top_k meilleurs chunks, avec leur rang dans chaque methode."""
        # 1) recherche par sens
        hits = vector_search(
            question, top_k=candidates, year_min=year_min,
            paper_id=paper_id, section=section, client=self.client,
        )
        by_meaning = [h.payload for h in hits]

        # 2) recherche par mots
        by_words = self.bm25_search(question, candidates, year_min, paper_id, section)

        # 3) fusion RRF : chaque liste donne 1/(60 + rang) points a ses chunks
        merged = {}  # cle = chunk_id -> un chunk present dans les deux listes n'apparait qu'une fois
        for label, chunks in (("vector_rank", by_meaning), ("bm25_rank", by_words)):
            for rank, chunk in enumerate(chunks, start=1):
                entry = merged.setdefault(
                    chunk["chunk_id"],
                    {**chunk, "score": 0.0, "vector_rank": None, "bm25_rank": None},
                )
                entry["score"] += 1.0 / (RRF_K + rank)
                entry[label] = rank

        results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
        return results[:top_k]


def show(results):
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title'][:60]} | page {r['page']} | {r['year']}")
        print(f"   rang sens={r['vector_rank']}  rang mots={r['bm25_rank']}  score={r['score']:.4f}")
        print(f"   {r['text'][:150]}...\n")


if __name__ == "__main__":
    # Demonstration : lancez  python -m rag.retriever   (apres python -m rag.indexing)
    retriever = HybridRetriever()

    q1 = "What are the limitations of machine learning for intrusion detection?"
    print(f"Question 1 : {q1}\n")
    show(retriever.search(q1, top_k=3))

    q2 = "HyDE hypothetical document embeddings"  # un nom propre : BM25 est utile ici
    print(f"Question 2 : {q2}\n")
    show(retriever.search(q2, top_k=3))

    retriever.close()