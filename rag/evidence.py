"""
rag/evidence.py - Etape 6 du RAG : l'Evidence Store (le "carnet de preuves").

Quand le retriever trouve un passage utile, on l'enregistre ici avec :
  - un identifiant court et stable : E001, E002, ...
  - sa source : article, DOI, section, page
  - son texte exact et son score de pertinence (de 0 a 1)

Les agents citeront ensuite "[E003]" dans leurs reponses. On peut alors
RETROUVER le passage exact, et DETECTER les citations inventees par le LLM.
"""
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rag.retriever import RRF_K

DB_PATH = "data/evidence.db"
MAX_SCORE = 2.0 / (RRF_K + 1)  # score maximal possible : 1er rang dans les DEUX methodes


@dataclass
class Evidence:
    evidence_id: str
    chunk_id: str
    paper_id: str
    title: str
    doi: Optional[str]
    section: str
    page: int
    text: str
    retrieval_score: float  # 0 a 1 (1 = premier dans les deux methodes de recherche)
    query: str  # la question qui a permis de trouver ce passage

    def to_dict(self):
        return asdict(self)

    def label(self):
        where = "resume" if self.section == "abstract" else f"page {self.page}"
        return f"{self.title} ({self.paper_id}, {where})"


class EvidenceStore:
    def __init__(self, db_path=DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS evidence (
                   num INTEGER PRIMARY KEY AUTOINCREMENT,
                   chunk_id TEXT UNIQUE,
                   paper_id TEXT, title TEXT, doi TEXT, section TEXT, page INTEGER,
                   text TEXT, retrieval_score REAL, query TEXT)"""
        )
        self.db.commit()

    def close(self):
        self.db.close()

    @staticmethod
    def _to_evidence(row):
        return Evidence(
            evidence_id=f"E{row['num']:03d}",
            chunk_id=row["chunk_id"], paper_id=row["paper_id"], title=row["title"],
            doi=row["doi"], section=row["section"], page=row["page"], text=row["text"],
            retrieval_score=row["retrieval_score"], query=row["query"],
        )

    def add(self, result, query):
        """Enregistre un resultat du retriever. Un meme passage garde toujours le meme identifiant."""
        score = round(min(result["score"] / MAX_SCORE, 1.0), 3)
        row = self.db.execute("SELECT num FROM evidence WHERE chunk_id = ?", (result["chunk_id"],)).fetchone()
        if row is None:
            cur = self.db.execute(
                """INSERT INTO evidence
                   (chunk_id, paper_id, title, doi, section, page, text, retrieval_score, query)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result["chunk_id"], result["paper_id"], result["title"], result.get("doi"),
                 result["section"], result["page"], result["text"], score, query),
            )
            num = cur.lastrowid
        else:
            num = row["num"]
            self.db.execute("UPDATE evidence SET retrieval_score = MAX(retrieval_score, ?) WHERE num = ?", (score, num))
        self.db.commit()
        return self.get(f"E{num:03d}")

    def get(self, evidence_id):
        """Retrouve une preuve a partir de son identifiant (ex : 'E003'). Renvoie None si inconnue."""
        match = re.fullmatch(r"E(\d+)", evidence_id)
        if not match:
            return None
        row = self.db.execute("SELECT * FROM evidence WHERE num = ?", (int(match.group(1)),)).fetchone()
        return self._to_evidence(row) if row else None

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]

    def collect(self, retriever, question, top_k=5, **filters):
        """Cherche avec le retriever ET enregistre les passages trouves comme preuves."""
        results = retriever.search(question, top_k=top_k, **filters)
        return [self.add(r, question) for r in results]

    def find_unknown_citations(self, text):
        """Repere les citations [E###] d'un texte qui n'existent PAS dans le carnet (citations inventees)."""
        cited = set(re.findall(r"\bE\d{3,}\b", text))
        return sorted(c for c in cited if self.get(c) is None)


def format_for_prompt(evidences):
    """Met les preuves en forme pour les donner a un LLM, avec leurs identifiants a citer."""
    blocks = [f"[{e.evidence_id}] {e.label()}\n{e.text}" for e in evidences]
    return "\n\n".join(blocks)


if __name__ == "__main__":
    # Demonstration : lancez  python -m rag.evidence   (apres python -m rag.indexing)
    from rag.retriever import HybridRetriever

    retriever = HybridRetriever()
    store = EvidenceStore()

    question = "What are the limitations of machine learning for intrusion detection?"
    print(f"Question : {question}\n")
    evidences = store.collect(retriever, question, top_k=3)
    print(format_for_prompt(evidences)[:900], "\n")
    print(f"Preuves dans le carnet : {store.count()}")
    for e in evidences:
        print(f"  {e.evidence_id}  score={e.retrieval_score}  {e.label()}")

    # Meme question : les identifiants ne changent pas, pas de doublons
    again = store.collect(retriever, question, top_k=3)
    print(f"\nApres la meme recherche : {store.count()} preuves (identiques : {[e.evidence_id for e in again]})")

    # Detection d'une citation inventee
    fake = "Selon [E001] et [E999], les faux positifs sont nombreux."
    print(f"\nTexte : {fake}")
    print(f"Citations inconnues : {store.find_unknown_citations(fake)}")

    retriever.close()
    store.close()