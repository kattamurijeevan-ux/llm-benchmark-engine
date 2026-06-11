from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = {
    "llama3-70b": "llama-3.3-70b-versatile",
    "llama3-8b":  "llama-3.1-8b-instant",
    "qwen":       "qwen/qwen3-32b"
}

SYSTEM_PROMPTS = {
    "medical":   "You are a medical expert. Answer the multiple choice question. Reply with only the answer letter (A, B, C, or D). Nothing else.",
    "math":      "You are a math expert. Solve step by step. Put your final numeric answer after '####'. Example: #### 42",
    "coding":    "You are a Python expert. Complete the function. Return only the function body code, no explanation.",
    "reasoning": "You are a reasoning expert. Answer with a short, direct answer. Be concise."
}

def run_single(model_key: str, question: str, domain: str) -> dict:
    model_id = MODELS.get(model_key, model_key)
    system = SYSTEM_PROMPTS.get(domain, "Answer the question concisely.")

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": question}
            ],
            max_tokens=512,
            temperature=0.0
        )
        latency_ms = (time.time() - start) * 1000
        return {
            "answer":     response.choices[0].message.content.strip(),
            "latency_ms": round(latency_ms, 2),
            "tokens":     response.usage.total_tokens,
            "error":      None
        }
    except Exception as e:
        return {"answer": "", "latency_ms": 0, "tokens": 0, "error": str(e)}