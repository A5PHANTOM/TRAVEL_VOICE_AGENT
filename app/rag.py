"""
Lightweight RAG using FAISS for vector search and scikit-learn TF-IDF for embeddings.

Why FAISS + TF-IDF?
- Zero model downloads at startup (no ONNX, no torch, no CUDA)
- Pure CPU, < 1 MB overhead beyond faiss-cpu and scikit-learn
- Index persists to disk and loads instantly on restart
"""

import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


class FaissRAG:
    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        persist_dir: str = "./faiss_db",
        chunk_size: int = 512,
        max_snippet_chars: int = 250,
    ):
        self.knowledge_dir = knowledge_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.max_snippet_chars = max_snippet_chars

        os.makedirs(self.persist_dir, exist_ok=True)

        self._index_path = os.path.join(self.persist_dir, "index.faiss")
        self._meta_path = os.path.join(self.persist_dir, "meta.pkl")

        self._vectorizer: TfidfVectorizer | None = None
        self._index: faiss.Index | None = None
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []

        # Load existing persisted index if available
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Load persisted index from disk. Returns True if successful."""
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            try:
                self._index = faiss.read_index(self._index_path)
                with open(self._meta_path, "rb") as f:
                    meta = pickle.load(f)
                self._vectorizer = meta["vectorizer"]
                self._documents = meta["documents"]
                self._metadatas = meta["metadatas"]
                return True
            except Exception:
                pass
        return False

    def _save(self):
        """Persist FAISS index and metadata to disk."""
        if self._index is not None:
            faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                },
                f,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_files(self) -> List[Dict]:
        """Walk knowledge_dir and return chunked text records."""
        docs = []
        for root, _, files in os.walk(self.knowledge_dir):
            for fn in sorted(files):
                if not fn.lower().endswith(".txt"):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                except Exception:
                    continue
                if not text:
                    continue
                for i in range(0, len(text), self.chunk_size):
                    chunk = text[i : i + self.chunk_size].strip()
                    if chunk:
                        docs.append({"text": chunk, "source": path})
        return docs

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_knowledge(self) -> int:
        """Read knowledge .txt files and build/rebuild the FAISS index.
        Returns the number of chunks indexed.
        """
        docs = self._read_files()
        if not docs:
            return 0

        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"]} for d in docs]

        # Fit TF-IDF; cap features to keep vectors compact
        vectorizer = TfidfVectorizer(max_features=4096, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)

        # L2-normalise for cosine similarity via inner product search
        matrix = self._l2_normalize(matrix)

        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = cosine on unit vectors
        index.add(matrix)

        self._vectorizer = vectorizer
        self._index = index
        self._documents = texts
        self._metadatas = metadatas

        self._save()
        return len(docs)

    def get_relevant(self, query: str, k: int = 3) -> List[Dict]:
        """Return top-k relevant documents for *query*.

        Each result dict: ``{"document": str, "metadata": dict, "score": float}``.
        """
        if not query or self._index is None or self._vectorizer is None:
            return []

        try:
            q_vec = self._vectorizer.transform([query]).toarray().astype(np.float32)
            q_vec = self._l2_normalize(q_vec)

            n_results = min(k, len(self._documents))
            scores, indices = self._index.search(q_vec, n_results)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = self._documents[idx]
                truncated = (
                    (doc[: self.max_snippet_chars] + "...")
                    if len(doc) > self.max_snippet_chars
                    else doc
                )
                results.append(
                    {
                        "document": truncated,
                        "metadata": self._metadatas[idx],
                        "score": float(score),
                    }
                )
            return results
        except Exception:
            return []
