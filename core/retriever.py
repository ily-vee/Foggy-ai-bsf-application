"""
retriever.py (v2)

Fixes vs the uploaded version:
- Loads per-chunk source metadata (filename, chunk_id) instead of bare text,
  so retrieved context can carry a real, checkable citation
  ("Source: bsf_manual.pdf") instead of the LLM having nothing to point to
  and inventing a fake DOI, which was the root cause chased down earlier in
  this project.
- Query embeddings are now normalized (normalize_embeddings=True) to match
  ingest.py's normalized corpus embeddings, so the dot product is a true
  cosine similarity rather than a magnitude-biased inner product.
- Added a minimum-relevance floor (MIN_RELEVANCE) computed on the RAW
  (pre-normalization) top similarity score. Min-max normalization alone
  cannot signal "nothing in the corpus matched" — the top result is always
  rescaled to ~1.0 regardless of how relevant it actually was. Without this,
  retrieve() would keep forcing top_k chunks into the prompt even for
  queries the corpus has no real answer to, which just re-creates the
  false-confidence problem from a different angle.
"""

import json
import numpy as np
from pathlib import Path

# NumPy 2.0 fix for rank_bm25
if not hasattr(np.ndarray, "ptp"):
    np.ndarray.ptp = lambda self, *args, **kwargs: np.ptp(self, *args, **kwargs)

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Below this raw cosine similarity, treat the corpus as having no relevant
# match for the query rather than forcing top_k chunks into the prompt anyway.
# Tune this against your own corpus/eval set — it's a reasonable starting
# point for all-MiniLM-L6-v2, not a universal constant.
MIN_RELEVANCE = 0.20


class HybridRetriever:
    def __init__(self, vector_db_dir="foggy_vector_db"):
        self.db_path = Path(vector_db_dir)

        chunks_file = self.db_path / "chunks.json"
        embeddings_file = self.db_path / "embeddings.npy"

        if not chunks_file.exists() or not embeddings_file.exists():
            raise FileNotFoundError(
                f"Vector DB files missing in '{self.db_path}'. "
                "Please run 'python ingest.py' first to build 'foggy_vector_db'."
            )

        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunk_records = json.load(f)

        # Back-compat: accept either the old bare-string list format or the
        # new metadata-carrying dict format, so an older chunks.json doesn't
        # hard-crash — it just won't have real source names to cite.
        if self.chunk_records and isinstance(self.chunk_records[0], str):
            self.chunk_records = [
                {"source": "unknown", "chunk_id": i, "text": t}
                for i, t in enumerate(self.chunk_records)
            ]

        self.texts = [r["text"] for r in self.chunk_records]
        self.embeddings = np.load(embeddings_file)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Tokenize corpus for BM25 Sparse Retrieval
        corpus = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(corpus)

        self.last_query_relevance = 0.0

    def retrieve(self, query: str, detected_stage: str = None, top_k: int = 3) -> str:
        """
        Retrieves top_k context chunks with source citations, formatted for
        direct insertion into the LLM prompt. Returns "" if nothing in the
        corpus clears the relevance floor for this query.

        Side effect: sets self.last_query_relevance to the raw top dense
        similarity score for this query, regardless of whether it cleared
        MIN_RELEVANCE. Callers can use this to distinguish "genuinely
        off-topic query" (very low relevance) from "on-topic but this small
        corpus doesn't cover it" (moderate relevance, still below the floor
        for confident grounding) — the two look identical if you only check
        whether retrieve() returned an empty string.
        """
        search_query = query
        if detected_stage:
            search_query = f"Black Soldier Fly {detected_stage} stage management. {query}"

        # 1. Dense Semantic Search (normalized vectors -> dot product == cosine sim)
        q_emb = self.embed_model.encode([search_query], convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = np.dot(self.embeddings, q_emb.T).flatten()

        top_raw_relevance = float(dense_scores.max()) if len(dense_scores) else 0.0
        self.last_query_relevance = top_raw_relevance
        if top_raw_relevance < MIN_RELEVANCE:
            return ""

        # 2. Sparse BM25 Search
        tokenized_query = search_query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 3. Score Normalization (for fusing two different score scales)
        norm_dense = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-8) if dense_scores.ptp() > 0 else dense_scores
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8) if bm25_scores.ptp() > 0 else bm25_scores

        # 4. Hybrid Weighted Fusion
        combined_scores = 0.5 * norm_dense + 0.5 * norm_bm25
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        retrieved_blocks = []
        for rank, idx in enumerate(top_indices, start=1):
            record = self.chunk_records[idx]
            # Source metadata is kept in self.chunk_records for your own
            # debugging/traceability, but deliberately NOT included in the
            # text handed to the LLM — the system prompt asks it to state
            # facts directly rather than cite anything, so there's no
            # reason to dangle a filename in front of it.
            retrieved_blocks.append(f"[{rank}] {record['text']}")

        return "\n\n".join(retrieved_blocks)