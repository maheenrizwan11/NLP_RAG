import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.generation import call_hf_api

CLAIM_EXTRACTION_PROMPT = """<s>[INST] Extract all factual claims from the following answer.
List each claim on a separate line, starting with "CLAIM:".
A claim is a single, atomic, verifiable statement of fact.
Do not include opinions or meta-statements.

Answer: {answer} [/INST]"""

CLAIM_VERIFICATION_PROMPT = """<s>[INST] You are a fact-checker. Given the context and a claim, determine if the claim is directly supported by the context.
Respond with exactly one word: SUPPORTED or NOT SUPPORTED.

Context: {context}

Claim: {claim} [/INST]"""

ALT_QUERY_PROMPT = """<s>[INST] Generate exactly 3 alternative questions that ask for the same information as the original question, worded differently.
Start each question with "Q:" on its own line. Nothing else.

Original Question: {question}
Answer: {answer} [/INST]"""


class LLMJudge:
    def __init__(self, hf_token, embedder):
        self.hf_token = hf_token
        self.embedder = embedder

    def extract_claims(self, answer):
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = call_hf_api(prompt, self.hf_token, max_new_tokens=300)
        claims = [
            line[len("CLAIM:"):].strip()
            for line in response.split('\n')
            if line.strip().upper().startswith("CLAIM:")
        ]
        return claims

    def verify_claim(self, claim, context):
        prompt = CLAIM_VERIFICATION_PROMPT.format(context=context[:2000], claim=claim)
        response = call_hf_api(prompt, self.hf_token, max_new_tokens=10)
        resp_upper = response.upper()
        return "SUPPORTED" in resp_upper and "NOT SUPPORTED" not in resp_upper

    def compute_faithfulness(self, answer, context):
        claims = self.extract_claims(answer)
        if not claims:
            return 1.0, {"claims": [], "verdicts": [], "score": 1.0, "note": "No claims extracted"}
        verdicts = [self.verify_claim(c, context) for c in claims]
        score = sum(verdicts) / len(verdicts)
        return score, {
            "claims": claims,
            "verdicts": ["SUPPORTED" if v else "NOT SUPPORTED" for v in verdicts],
            "score": round(score, 3)
        }

    def generate_alt_queries(self, question, answer):
        prompt = ALT_QUERY_PROMPT.format(question=question, answer=answer)
        response = call_hf_api(prompt, self.hf_token, max_new_tokens=200)
        queries = [
            line[2:].strip()
            for line in response.split('\n')
            if line.strip().upper().startswith("Q:")
        ]
        return queries[:3]

    def compute_relevancy(self, question, answer):
        alt_queries = self.generate_alt_queries(question, answer)
        if not alt_queries:
            return 0.5, {"alt_queries": [], "similarities": [], "score": 0.5,
                         "note": "No alt queries generated"}
        orig_emb = self.embedder.encode([question])
        alt_embs = self.embedder.encode(alt_queries)
        sims = cosine_similarity(orig_emb, alt_embs)[0].tolist()
        score = float(np.mean(sims))
        return score, {
            "alt_queries": alt_queries,
            "similarities": [round(s, 3) for s in sims],
            "score": round(score, 3)
        }

    def evaluate_single(self, question, answer, context):
        faith_score, faith_detail = self.compute_faithfulness(answer, context)
        rel_score, rel_detail = self.compute_relevancy(question, answer)
        return {
            "question": question,
            "answer": answer,
            "faithfulness": round(faith_score, 3),
            "relevancy": round(rel_score, 3),
            "faithfulness_detail": faith_detail,
            "relevancy_detail": rel_detail
        }
