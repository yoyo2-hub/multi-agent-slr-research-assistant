"""
rag/indexing.py - Etape 4 du RAG : ranger les chunks dans Qdrant.

Qdrant est une base de donnees faite pour retrouver des vecteurs proches.
Ici on l'utilise en mode LOCAL : tout est stocke dans le dossier data/qdrant,
pas de serveur ni de Docker a installer.

Pour chaque chunk, Qdrant garde :
  - son vecteur (les 384 nombres de l'etape 3)
  - sa "payload" : le texte + titre, page, section, annee, DOI... (les metadonnees)
"""
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from rag.embeddings import embed_query, embed_texts

QDRANT_PATH = "data/qdrant"
COLLECTION = "scientific_chunks"
VECTOR_SIZE = 384  # taille des embeddings du modele all-MiniLM-L6-v2


def get_client():
    return QdrantClient(path=QDRANT_PATH)


def build_index(chunks_json="data/chunks.json", batch_size=256):
    """Lit data/chunks.json, calcule les embeddings et remplit Qdrant (reconstruit l'index a zero)."""
    chunks = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
    client = get_client()

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embed_texts([c["text"] for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),  # identifiant stable
                vector=vec,
                payload=c,  # tout le chunk (texte + metadonnees) est conserve
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(COLLECTION, points=points)
        print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)} chunks indexes")

    return client


def vector_search(question, top_k=5, year_min=None, paper_id=None, section=None, client=None):
    """Cherche les chunks dont le SENS est le plus proche de la question.

    Filtres facultatifs : annee minimale, un article precis, une section precise.
    """
    conditions = []
    if year_min is not None:
        conditions.append(FieldCondition(key="year", range=Range(gte=year_min)))
    if paper_id is not None:
        conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))
    if section is not None:
        conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
    query_filter = Filter(must=conditions) if conditions else None

    own_client = client is None
    client = client or get_client()
    hits = client.query_points(
        COLLECTION,
        query=embed_query(question),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points
    if own_client:
        client.close()
    return hits


def show(hits):
    for rank, h in enumerate(hits, 1):
        p = h.payload
        print(f"{rank}. score={h.score:.2f} | {p['title'][:60]} | page {p['page']} | {p['year']}")
        print(f"   {p['text'][:150]}...\n")


if __name__ == "__main__":
    # 1) construire l'index
    client = build_index()
    print(f"Index pret : {client.count(COLLECTION).count} chunks dans Qdrant\n")

    # 2) test : poser une question (en ANGLAIS, comme les articles)
    question = "What are the limitations of machine learning for intrusion detection?"
    print(f"Question : {question}\n")
    show(vector_search(question, top_k=3, client=client))

    # 3) test des filtres : uniquement les articles de 2023 ou plus recents
    print("--- Meme question, articles depuis 2023 seulement ---\n")
    show(vector_search(question, top_k=3, year_min=2023, client=client))

    client.close()