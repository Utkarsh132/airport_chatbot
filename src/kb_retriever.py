"""
Knowledge base retrieval engine.

Builds:
1. A text embedding index over KB descriptions (service_name + short_description
   + category) for semantic retrieval from text/voice queries.
2. An image embedding index (FAISS if available, else brute-force cosine)
   over the synthetic image manifest, mapped to KB categories, for
   image-based retrieval.

Retrieval strategy:
- Text query -> embed -> cosine/FAISS search over KB text index -> top-k KB records.
- Image query -> embed -> cosine/FAISS search over image index -> majority category
  among top-k -> map to representative KB record(s) of that category.
- Entity exact-match (e.g., gate number) boosts the corresponding KB record's score.
"""

import os
import numpy as np

from config import TOP_K_RETRIEVAL
from data_loader import load_knowledge_base, build_image_manifest
from text_pipeline import embed_text
from image_pipeline import embed_image
from utils import cosine_similarity

_FAISS_AVAILABLE = True
try:
    import faiss  # noqa: F401
except Exception:
    _FAISS_AVAILABLE = False


class KnowledgeBaseRetriever:
    def __init__(self):
        self.kb_df = load_knowledge_base()
        self.image_manifest = build_image_manifest()

        self._text_index = None
        self._text_vectors = None
        self._image_index = None
        self._image_vectors = None

        self._build_text_index()
        self._build_image_index()

    # ------------------------------------------------------------------
    # TEXT INDEX
    # ------------------------------------------------------------------
    def _kb_text_repr(self, row) -> str:
        return f"{row['service_name']}. {row['category']}. {row['short_description']}"

    def _build_text_index(self):
        corpus = [self._kb_text_repr(row) for _, row in self.kb_df.iterrows()]
        vectors = np.stack([embed_text(t, corpus_for_tfidf=corpus) for t in corpus])
        self._text_vectors = vectors.astype(np.float32)

        if _FAISS_AVAILABLE:
            dim = self._text_vectors.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ~ cosine
            index.add(self._text_vectors)
            self._text_index = index

    def query_text(self, text: str, top_k: int = TOP_K_RETRIEVAL):
        query_vec = embed_text(text, corpus_for_tfidf=[self._kb_text_repr(r) for _, r in self.kb_df.iterrows()])
        query_vec = query_vec.astype(np.float32)

        if _FAISS_AVAILABLE and self._text_index is not None and query_vec.shape[0] == self._text_vectors.shape[1]:
            scores, idxs = self._text_index.search(query_vec.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                record = self.kb_df.iloc[idx].to_dict()
                results.append({"record": record, "score": float(score)})
            return results

        # brute-force cosine fallback
        sims = [cosine_similarity(query_vec, v) for v in self._text_vectors]
        order = np.argsort(sims)[::-1][:top_k]
        return [{"record": self.kb_df.iloc[i].to_dict(), "score": float(sims[i])} for i in order]

    # ------------------------------------------------------------------
    # IMAGE INDEX
    # ------------------------------------------------------------------
    def _build_image_index(self):
        if len(self.image_manifest) == 0:
            self._image_vectors = np.zeros((0, 1), dtype=np.float32)
            return

        vectors = np.stack([embed_image(p) for p in self.image_manifest["path"].tolist()])
        self._image_vectors = vectors.astype(np.float32)

        if _FAISS_AVAILABLE:
            dim = self._image_vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._image_vectors)
            self._image_index = index

    def query_image(self, path_or_image, top_k: int = TOP_K_RETRIEVAL):
        if len(self.image_manifest) == 0:
            return []

        query_vec = embed_image(path_or_image).astype(np.float32)

        if _FAISS_AVAILABLE and self._image_index is not None and query_vec.shape[0] == self._image_vectors.shape[1]:
            scores, idxs = self._image_index.search(query_vec.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                row = self.image_manifest.iloc[idx].to_dict()
                results.append({"category": row["category"], "path": row["path"], "score": float(score)})
            return results

        sims = [cosine_similarity(query_vec, v) for v in self._image_vectors]
        order = np.argsort(sims)[::-1][:top_k]
        return [{"category": self.image_manifest.iloc[i]["category"],
                  "path": self.image_manifest.iloc[i]["path"],
                  "score": float(sims[i])} for i in order]

    def kb_records_for_category(self, category: str):
        matches = self.kb_df[self.kb_df["category"] == category]
        return matches.to_dict("records")

    def kb_record_by_id(self, kb_id: str):
        matches = self.kb_df[self.kb_df["id"] == kb_id]
        if len(matches):
            return matches.iloc[0].to_dict()
        return None

    def kb_record_by_gate(self, gate: str):
        gate_norm = gate.upper().replace(" ", "").replace("-", "")
        for _, row in self.kb_df.iterrows():
            if gate_norm in row["service_name"].upper().replace(" ", ""):
                return row.to_dict()
        return None


if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever()
    results = retriever.query_text("Where is gate B12?")
    for r in results:
        print(r["record"]["service_name"], round(r["score"], 3))
