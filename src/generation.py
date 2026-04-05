import os
import time

GROQ_MODEL = "llama-3.1-8b-instant"

ANSWER_PROMPT = """You are a precise question-answering assistant.
Answer the question using ONLY the information provided in the context below.
If the context does not contain enough information, say "I cannot answer this from the provided context."

Context:
{context}

Question: {question}"""


def call_hf_api(prompt, hf_token=None, max_new_tokens=300, temperature=0.1, retries=3):
    import groq as groq_lib

    content = prompt
    for tag in ('<s>', '[INST]', '[/INST]', '</s>'):
        content = content.replace(tag, '')
    content = content.strip()

    client = groq_lib.Groq(api_key=os.environ["GROQ_API_KEY"])
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_new_tokens,
                temperature=max(temperature, 1e-6),
            )
            return resp.choices[0].message.content.strip()
        except groq_lib.RateLimitError:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise
    return ""


def generate_answer(question, context_chunks, hf_token=None):
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks)])
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    answer = call_hf_api(prompt, hf_token)
    return answer, context
