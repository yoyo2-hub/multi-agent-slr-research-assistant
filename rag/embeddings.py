
"""
rag/embeddings.py - Etape 3 du RAG : transformer un texte en nombres (embedding).

Un embedding = une liste de 384 nombres qui represente le SENS d'un texte.
Deux textes qui parlent de la meme chose ont des embeddings proches.

IMPORTANT : on utilise le MEME modele pour les chunks et pour les questions.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # petit, rapide, marche sans GPU
_model = None


def get_model():
    """Charge le modele une seule fois (le premier lancement le telecharge, ~90 Mo)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=32):
    """Transforme une liste de textes en une liste de vecteurs (384 nombres chacun)."""
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # vecteurs de longueur 1 : le produit scalaire = la ressemblance
    )
    return vectors.tolist()


def embed_query(question):
    """Transforme UNE question en vecteur."""
    return embed_texts([question])[0]


def similarity(vec_a, vec_b):
    """Ressemblance entre deux vecteurs normalises : proche de 1 = meme sens, proche de 0 = sans rapport."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


if __name__ == "__main__":
    # Petite demonstration : lancez  python -m rag.embeddings
    phrases = {
        "A": "Machine learning methods for network intrusion detection",
        "B": "Detecting cyber attacks on computer networks with neural networks",
        "C": "A recipe for chocolate cake with butter and sugar",
    }
    vecs = {k: embed_query(v) for k, v in phrases.items()}

    print(f"Taille d'un embedding : {len(vecs['A'])} nombres")
    print(f"Debut de l'embedding A : {[round(x, 3) for x in vecs['A'][:5]]} ...\n")
    for k, v in phrases.items():
        print(f"{k} : {v}")
    print()
    print(f"A vs B (meme sujet)        : {similarity(vecs['A'], vecs['B']):.2f}")
    print(f"A vs C (sujets differents) : {similarity(vecs['A'], vecs['C']):.2f}")