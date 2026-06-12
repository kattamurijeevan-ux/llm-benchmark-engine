from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from app.models import BenchmarkRunRequest, CompareRequest
from app.database import SessionLocal, RunRecord
from evaluation.pipeline import run_benchmark

app = FastAPI(
    title="LLM Benchmark Engine",
    version="1.0.0",
    description="Measures LLM accuracy, hallucination rate, and latency across medical, math, coding, and reasoning domains."
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")
@app.get("/")
def root():
    return {
        "service": "LLM Benchmark Engine",
        "models":  ["llama3-70b", "llama3-8b", "qwen"],
        "domains": ["medical", "math", "coding", "reasoning"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/benchmark")
def run_single_benchmark(req: BenchmarkRunRequest):
    try:
        return run_benchmark(req.model, req.domain, req.num_samples)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
def compare_models(req: CompareRequest):
    results = []
    for model in req.models:
        result = run_benchmark(model, req.domain, req.num_samples)
        results.append(result)
    results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
    return {
        "domain":            req.domain,
        "samples_per_model": req.num_samples,
        "ranking":           results,
        "winner":            results[0]["model"] if results else None
    }

@app.get("/results")
def get_all_results():
    db   = SessionLocal()
    runs = db.query(RunRecord).order_by(RunRecord.created_at.desc()).limit(50).all()
    db.close()
    return [
        {
            "run_id":             r.id,
            "model":              r.model,
            "domain":             r.domain,
            "accuracy":           r.accuracy,
            "hallucination_rate": r.hallucination_rate,
            "avg_latency_ms":     r.avg_latency_ms,
            "samples_tested":     r.samples_tested,
            "created_at":         str(r.created_at)
        }
        for r in runs
    ]

@app.get("/results/{domain}")
def get_results_by_domain(domain: str):
    db   = SessionLocal()
    runs = db.query(RunRecord)\
             .filter(RunRecord.domain == domain)\
             .order_by(RunRecord.accuracy.desc())\
             .all()
    db.close()
    return [
        {
            "model":              r.model,
            "accuracy":           r.accuracy,
            "hallucination_rate": r.hallucination_rate,
            "avg_latency_ms":     r.avg_latency_ms,
            "samples_tested":     r.samples_tested
        }
        for r in runs
    ]