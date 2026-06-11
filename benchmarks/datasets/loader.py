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
    ds = load_dataset(
        "bigbio/med_qa",
        "med_qa_en_source",
        split="test",
        trust_remote_code=True
    )
    samples = random.sample(list(ds), min(n, len(ds)))
    results = []
    for s in samples:
        options = s.get("options", {})
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        results.append({
            "question": f"{s['question']}\n\nOptions:\n{options_text}",
            "answer": s["answer_idx"],
            "domain": "medical"
        })
    return results

def _load_gsm8k(n: int) -> list[dict]:
    ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
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
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
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
    ds = load_dataset(
        "hotpot_qa",
        "fullwiki",
        split="validation",
        trust_remote_code=True
    )
    samples = random.sample(list(ds), min(n, len(ds)))
    return [
        {
            "question": s["question"],
            "answer": s["answer"],
            "domain": "reasoning"
        }
        for s in samples
    ]