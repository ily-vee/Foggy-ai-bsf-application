import json
import numpy as np
from pathlib import Path

# NumPy 2.0 fix for rank_bm25
if not hasattr(np.ndarray, "ptp"):
    np.ndarray.ptp = lambda self, *args, **kwargs: np.ptp(self, *args, **kwargs)

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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
            self.chunks = json.load(f)
            
        self.embeddings = np.load(embeddings_file)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Tokenize corpus for BM25 Sparse Retrieval
        corpus = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, detected_stage: str = None, top_k: int = 3) -> str:
        """
        Retrieves top_k context chunks. 
        If a vision-detected life stage is passed, it injects it into the search query.
        """
        search_query = query
        if detected_stage:
            search_query = f"Black Soldier Fly {detected_stage} stage management. {query}"

        # 1. Dense Semantic Search
        q_emb = self.embed_model.encode([search_query], convert_to_numpy=True)
        dense_scores = np.dot(self.embeddings, q_emb.T).flatten()

        # 2. Sparse BM25 Search
        tokenized_query = search_query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 3. Score Normalization
        norm_dense = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-8) if dense_scores.ptp() > 0 else dense_scores
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8) if bm25_scores.ptp() > 0 else bm25_scores

        # 4. Hybrid Reciprocal Rank / Weighted Fusion
        combined_scores = 0.5 * norm_dense + 0.5 * norm_bm25
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        retrieved_blocks = []
        for rank, idx in enumerate(top_indices, start=1):
            retrieved_blocks.append(f"[{rank}] {self.chunks[idx]}")

        return "\n\n".join(retrieved_blocks)