import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from src.retrieval import HybridRetriever
from src.generation import generate_answer
from src.evaluation import LLMJudge

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
INDEX_NAME = "rag-qa-nlp"

CONFIGS = [
    {"name": "fixed_semantic",     "namespace": "fixed",     "bm25": "app/bm25_fixed.pkl",     "mode": "semantic"},
    {"name": "fixed_hybrid",       "namespace": "fixed",     "bm25": "app/bm25_fixed.pkl",     "mode": "hybrid_rerank"},
    {"name": "recursive_semantic", "namespace": "recursive", "bm25": "app/bm25_recursive.pkl", "mode": "semantic"},
    {"name": "recursive_hybrid",   "namespace": "recursive", "bm25": "app/bm25_recursive.pkl", "mode": "hybrid_rerank"},
]


def load_queries(path):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def run_config(config, queries, embedder):
    print(f"\nConfig: {config['name']}")

    retriever = HybridRetriever(
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        namespace=config["namespace"],
        bm25_pkl_path=config["bm25"]
    )
    judge = LLMJudge(hf_token=HF_TOKEN, embedder=embedder)

    results = []
    for q in queries:
        print(f"Q{q['query_id']}: {q['question'][:70]}")
        chunks = retriever.retrieve(q["question"], mode=config["mode"])
        answer, context = generate_answer(q["question"], chunks, HF_TOKEN)
        time.sleep(1)

        eval_result = judge.evaluate_single(q["question"], answer, context)
        eval_result["query_id"] = q["query_id"]
        eval_result["config"] = config["name"]
        results.append(eval_result)

        print(f"  Faithfulness: {eval_result['faithfulness']:.3f} | Relevancy: {eval_result['relevancy']:.3f}")
        time.sleep(1)

    out_path = f"eval/results_{config['name']}.json"
    os.makedirs("eval", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    queries = load_queries("eval/test_queries.jsonl")
    print(f"Loaded {len(queries)} queries")

    all_results = {}
    for config in CONFIGS:
        all_results[config["name"]] = run_config(config, queries, embedder)

    print("\nResults:")
    for name, results in all_results.items():
        avg_f = sum(r["faithfulness"] for r in results) / len(results)
        avg_r = sum(r["relevancy"] for r in results) / len(results)
        print(f"  {name}: F={avg_f:.3f} R={avg_r:.3f}")
