from benchmarks.datasets.loader import load_samples
from benchmarks.runners.groq_runner import run_single
from benchmarks.runners.scorer import score
from app.database import SessionLocal, RunRecord, SampleRecord
from tqdm import tqdm
import json
import uuid

def run_benchmark(model: str, domain: str, num_samples: int = 20) -> dict:
    samples  = load_samples(domain, num_samples)
    run_id   = str(uuid.uuid4())
    db       = SessionLocal()

    correct       = 0
    hallucinations = 0
    total_latency  = 0
    total_tokens   = 0
    results        = []

    print(f"\nRunning {model} on {domain} ({len(samples)} samples)...")

    for s in tqdm(samples):
        output = run_single(model, s["question"], domain)

        if output["error"]:
            print(f"  Error: {output['error']}")
            continue

        scores = score(output["answer"], s["answer"], domain)

        if scores["is_correct"]:      correct += 1
        if scores["is_hallucination"]: hallucinations += 1

        total_latency += output["latency_ms"]
        total_tokens  += output["tokens"]

        db.add(SampleRecord(
            run_id         = run_id,
            model          = model,
            domain         = domain,
            question       = s["question"][:1000],
            correct_answer = str(s["answer"])[:500],
            model_answer   = output["answer"][:500],
            is_correct     = int(scores["is_correct"]),
            is_hallucination = int(scores["is_hallucination"]),
            latency_ms     = output["latency_ms"],
            tokens_used    = output["tokens"]
        ))
        results.append(scores)

    n = len(results)
    if n == 0:
        db.close()
        return {"error": "No results — check your Groq API key"}

    accuracy          = round(correct / n, 4)
    hallucination_rate = round(hallucinations / n, 4)
    avg_latency        = round(total_latency / n, 2)

    db.add(RunRecord(
        id                 = run_id,
        model              = model,
        domain             = domain,
        accuracy           = accuracy,
        hallucination_rate = hallucination_rate,
        avg_latency_ms     = avg_latency,
        total_tokens       = total_tokens,
        samples_tested     = n,
        raw_results        = json.dumps(results)
    ))
    db.commit()
    db.close()

    return {
        "run_id":             run_id,
        "model":              model,
        "domain":             domain,
        "accuracy":           accuracy,
        "hallucination_rate": hallucination_rate,
        "avg_latency_ms":     avg_latency,
        "total_tokens":       total_tokens,
        "samples_tested":     n
    }