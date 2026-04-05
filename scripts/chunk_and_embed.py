import os
import json
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "rag-qa-nlp"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)


def chunk_fixed(text, doc_id, chunk_size=512, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    idx = 0
    i = 0
    while i < len(tokens):
        chunk_text = enc.decode(tokens[i: i + chunk_size]).strip()
        if len(chunk_text) > 30:
            chunks.append({
                "id": f"{doc_id}_fixed_{idx}",
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_index": idx,
                "strategy": "fixed",
            })
            idx += 1
        i += chunk_size - overlap
    return chunks


def chunk_recursive(text, doc_id, chunk_size=600, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(text)
    return [
        {
            "id": f"{doc_id}_recursive_{i}",
            "text": t.strip(),
            "doc_id": doc_id,
            "chunk_index": i,
            "strategy": "recursive",
        }
        for i, t in enumerate(splits)
        if len(t.strip()) > 30
    ]


def load_articles():
    articles = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8") as f:
            content = f.read()
        articles.append({"doc_id": fname.replace(".txt", ""), "text": content})
    return articles


def embed_and_upsert(chunks, model, index, namespace):
    print(f"Upserting {len(chunks)} chunks to namespace '{namespace}'")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeds = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        vectors = [
            {
                "id": c["id"],
                "values": emb.tolist(),
                "metadata": {k: v for k, v in c.items() if k != "id"},
            }
            for c, emb in zip(batch, embeds)
        ]
        index.upsert(vectors=vectors, namespace=namespace)


def save_jsonl(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    print("Loading model...")
    model = SentenceTransformer(EMBED_MODEL)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    all_fixed, all_recursive = [], []
    for art in tqdm(articles, desc="Chunking"):
        all_fixed += chunk_fixed(art["text"], art["doc_id"])
        all_recursive += chunk_recursive(art["text"], art["doc_id"])

    print(f"Fixed: {len(all_fixed)} | Recursive: {len(all_recursive)}")

    save_jsonl(all_fixed, os.path.join(PROC_DIR, "chunks_fixed.jsonl"))
    save_jsonl(all_recursive, os.path.join(PROC_DIR, "chunks_recursive.jsonl"))

    embed_and_upsert(all_fixed, model, index, "fixed")
    embed_and_upsert(all_recursive, model, index, "recursive")

    print(index.describe_index_stats())
