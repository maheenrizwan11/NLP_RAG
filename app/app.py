import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import nltk
import gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.retrieval import HybridRetriever
from src.generation import generate_answer
from src.evaluation import LLMJudge

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BM25_FIXED = os.path.join(BASE_DIR, "app", "bm25_fixed.pkl")
BM25_RECURSIVE = os.path.join(BASE_DIR, "app", "bm25_recursive.pkl")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "rag-qa-nlp"


def _build_bm25(chunks_path, pkl_path):
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    corpus = [c["text"] for c in chunks]
    tokenized = [nltk.word_tokenize(t.lower()) for t in corpus]
    bm25 = BM25Okapi(tokenized)
    data = {
        "bm25": bm25,
        "corpus": corpus,
        "ids": [c["id"] for c in chunks],
        "metadata": [{k: v for k, v in c.items() if k != "id"} for c in chunks],
    }
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


if not os.path.exists(BM25_FIXED):
    _build_bm25(os.path.join(BASE_DIR, "data", "processed", "chunks_fixed.jsonl"), BM25_FIXED)
if not os.path.exists(BM25_RECURSIVE):
    _build_bm25(os.path.join(BASE_DIR, "data", "processed", "chunks_recursive.jsonl"), BM25_RECURSIVE)

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

RETRIEVERS = {
    "fixed": HybridRetriever(PINECONE_API_KEY, INDEX_NAME, "fixed", BM25_FIXED),
    "recursive": HybridRetriever(PINECONE_API_KEY, INDEX_NAME, "recursive", BM25_RECURSIVE),
}

JUDGE = LLMJudge(hf_token=None, embedder=EMBEDDER)

MODE_MAP = {
    "Semantic Only": "semantic",
    "Hybrid + Reranking": "hybrid_rerank",
}
STRATEGY_MAP = {
    "Fixed 512-token": "fixed",
    "Recursive Character": "recursive",
}


def handle_ask(question, mode_label, strategy_label, state):
    if not question.strip():
        return "Please enter a question.", "", state

    mode = MODE_MAP[mode_label]
    strategy = STRATEGY_MAP[strategy_label]
    retriever = RETRIEVERS[strategy]

    chunks = retriever.retrieve(question, mode=mode)
    answer, context = generate_answer(question, chunks)

    context_display = "\n\n---\n\n".join(
        f"**[{i+1}]** {c['text'][:500]}" for i, c in enumerate(chunks)
    )

    new_state = {"question": question, "answer": answer, "context": context}
    return answer, context_display, new_state


def handle_evaluate(state):
    if not state or not state.get("answer"):
        return 0.0, 0.0, {"error": "Ask a question first in Tab 1"}
    result = JUDGE.evaluate_single(state["question"], state["answer"], state["context"])
    return result["faithfulness"], result["relevancy"], result


def load_ablation_data():
    configs = ["fixed_semantic", "fixed_hybrid", "recursive_semantic", "recursive_hybrid"]
    rows = []
    for cfg in configs:
        path = os.path.join(EVAL_DIR, f"results_{cfg}.json")
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            avg_f = sum(r["faithfulness"] for r in results) / len(results)
            avg_r = sum(r["relevancy"] for r in results) / len(results)
            rows.append([cfg, f"{avg_f:.3f}", f"{avg_r:.3f}", f"{(avg_f+avg_r)/2:.3f}"])
        else:
            rows.append([cfg, "N/A", "N/A", "N/A"])
    return rows


with gr.Blocks(title="RAG QA System") as demo:
    gr.Markdown("# RAG QA System")
    state = gr.State({})

    with gr.Tabs():
        with gr.Tab("Ask"):
            with gr.Row():
                mode_dd = gr.Dropdown(
                    choices=["Semantic Only", "Hybrid + Reranking"],
                    value="Hybrid + Reranking",
                    label="Retrieval mode",
                )
                strategy_dd = gr.Dropdown(
                    choices=["Fixed 512-token", "Recursive Character"],
                    value="Fixed 512-token",
                    label="Chunking",
                )
            question_box = gr.Textbox(label="Question", lines=2)
            ask_btn = gr.Button("Submit")
            answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
            with gr.Accordion("Context chunks", open=False):
                context_md = gr.Markdown()
            ask_btn.click(
                handle_ask,
                inputs=[question_box, mode_dd, strategy_dd, state],
                outputs=[answer_box, context_md, state],
            )

        with gr.Tab("Evaluate"):
            gr.Markdown("Evaluate the last answer (faithfulness + relevancy).")
            eval_btn = gr.Button("Run")
            with gr.Row():
                faith_num = gr.Number(label="Faithfulness", precision=3)
                rel_num = gr.Number(label="Relevancy", precision=3)
            detail_json = gr.JSON(label="Details")
            eval_btn.click(
                handle_evaluate,
                inputs=[state],
                outputs=[faith_num, rel_num, detail_json],
            )

        with gr.Tab("Ablation"):
            gr.Dataframe(
                value=load_ablation_data(),
                headers=["Config", "Faithfulness", "Relevancy", "Combined"],
                interactive=False,
            )

if __name__ == "__main__":
    demo.launch()
