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
    # Clean parquet MedQA dataset, no loading script
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    samples = random.sample(list(ds), min(n, len(ds)))
    results = []
    for s in samples:
        options_text = (
            f"A: {s['opa']}\n"
            f"B: {s['opb']}\n"
            f"C: {s['opc']}\n"
            f"D: {s['opd']}"
        )
        correct = ["A", "B", "C", "D"][s["cop"]]
        results.append({
            "question": f"{s['question']}\n\nOptions:\n{options_text}",
            "answer": correct,
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
    # All hotpotqa/trivia_qa forks use blocked loading scripts.
    # lucadiliello/triviaqa is clean parquet, no loading script needed.
    ds = load_dataset("lucadiliello/triviaqa", split="train")
    samples = random.sample(list(ds), min(n, len(ds)))
    return [
        {
            "question": s["question"],
            "answer": s["answers"][0] if isinstance(s["answers"], list) else s["answers"],
            "domain": "reasoning"
        }
        for s in samples
    ]