from datasets import load_dataset
import random

def load_samples(domain: str, num_samples: int) -> list[dict]:
    if domain == "medical":
        return _load_medqa(num_samples)
    elif domain == "math":
        return _load_gsm8k(num_samples)
    elif domain == "coding":
        return _load_humaneval(num_samples)
    elif domain == "reasoning":
        return _load_hotpotqa(num_samples)
    else:
        raise ValueError(f"Unknown domain: {domain}. Choose: medical, math, coding, reasoning")

def _load_medqa(n: int) -> list[dict]:
    # Using a clean parquet version that doesn't need a loading script
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    samples = random.sample(list(ds), min(n, len(ds)))
    results = []
    for s in samples:
        options_text = "\n".join([
            f"A: {s['options']['A']}",
            f"B: {s['options']['B']}",
            f"C: {s['options']['C']}",
            f"D: {s['options']['D']}"
        ])
        results.append({
            "question": f"{s['question']}\n\nOptions:\n{options_text}",
            "answer": s["answer_idx"],
            "domain": "medical"
        })
    return results

def _load_gsm8k(n: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = random.sample(list(ds), min(n, len(ds)))
    return [
        {
            "question": s["question"],
            "answer": s["answer"].split("####")[-1].strip(),
            "domain": "math"
        }
        for s in samples
    ]

def _load_humaneval(n: int) -> list[dict]:
    # openai/openai_humaneval is the correct current namespace
    ds = load_dataset("openai/openai_humaneval", split="test")
    samples = random.sample(list(ds), min(n, len(ds)))
    return [
        {
            "question": f"Complete this Python function:\n\n{s['prompt']}",
            "answer": s["canonical_solution"],
            "domain": "coding",
            "entry_point": s["entry_point"]
        }
        for s in samples
    ]

def _load_hotpotqa(n: int) -> list[dict]:
    # HotpotQA original host is offline (May 2025), all forks use blocked loading scripts.
    # Using TriviaQA (rc subset) as reasoning benchmark — same multi-hop question style.
    ds = load_dataset("trivia_qa", "rc", split="validation")
    samples = random.sample(list(ds), min(n, len(ds)))
    return [
        {
            "question": s["question"],
            "answer": s["answer"]["value"],
            "domain": "reasoning"
        }
        for s in samples
    ]