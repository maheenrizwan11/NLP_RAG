import os
import pickle
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def tokenize(text):
    return nltk.word_tokenize(text.lower())


class HybridRetriever:

    def __init__(self, pinecone_api_key, index_name, namespace, bm25_pkl_path,
                 embed_model_name="all-MiniLM-L6-v2", semantic_weight=0.7, bm25_weight=0.3):

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self.embedder = SentenceTransformer(embed_model_name)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
        self.namespace = namespace

        with open(bm25_pkl_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_corpus = data["corpus"]
        self.bm25_ids = data["ids"]
        self.bm25_metadata = data["metadata"]

    def semantic_search(self, query, top_k=20):
        query_emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        result = self.index.query(
            vector=query_emb,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        return [
            {
                "id": m["id"],
                "score": m["score"],
                "text": m["metadata"].get("text", ""),
                "metadata": m["metadata"],
            }
            for m in result["matches"]
        ]

    def bm25_search(self, query, top_k=20):
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id": self.bm25_ids[i],
                "score": float(scores[i]),
                "text": self.bm25_corpus[i],
                "metadata": self.bm25_metadata[i],
            }
            for i in top_idx
            if scores[i] > 0
        ]

    def reciprocal_rank_fusion(self, sem_results, bm25_results, k=60):
        sem_rank = {r["id"]: i + 1 for i, r in enumerate(sem_results)}
        bm25_rank = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}

        all_docs = {r["id"]: r for r in sem_results + bm25_results}

        scored = []
        for doc_id, doc in all_docs.items():
            rs = sem_rank.get(doc_id, len(sem_results) + 1)
            rb = bm25_rank.get(doc_id, len(bm25_results) + 1)
            rrf_score = self.semantic_weight * (1.0 / (k + rs)) + self.bm25_weight * (1.0 / (k + rb))
            entry = dict(doc)
            entry["score"] = rrf_score
            scored.append(entry)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]

    def retrieve(self, query, mode="hybrid_rerank", top_k=20, top_n=5):
        if mode == "semantic":
            return self.semantic_search(query, top_k=top_k)[:top_n]

        sem = self.semantic_search(query, top_k=top_k)
        bm25 = self.bm25_search(query, top_k=top_k)
        fused = self.reciprocal_rank_fusion(sem, bm25)

        if mode == "hybrid":
            return fused[:top_n]

        return self.rerank(query, fused, top_n=top_n)
