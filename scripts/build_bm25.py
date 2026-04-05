import os
import json
import pickle
import nltk
from rank_bm25 import BM25Okapi
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

PROC_DIR = "data/processed"
APP_DIR = "app"
os.makedirs(APP_DIR, exist_ok=True)


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def build_bm25(jsonl_path, out_path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Building BM25 for {len(chunks)} chunks...")

    corpus = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "id"} for c in chunks]

    tokenized = [tokenize(t) for t in tqdm(corpus, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)

    payload = {"bm25": bm25, "corpus": corpus, "ids": ids, "metadata": metadata}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    build_bm25(
        os.path.join(PROC_DIR, "chunks_fixed.jsonl"),
        os.path.join(APP_DIR, "bm25_fixed.pkl"),
    )
    build_bm25(
        os.path.join(PROC_DIR, "chunks_recursive.jsonl"),
        os.path.join(APP_DIR, "bm25_recursive.pkl"),
    )
