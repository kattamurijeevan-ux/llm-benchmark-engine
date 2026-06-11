# LLM Benchmark Engine

Measures LLM accuracy, hallucination rate, and latency across four domains using real public datasets.

## Models tested
- Llama 3.1 70B (Groq)
- Mixtral 8x7B (Groq)
- Gemma 7B (Groq)

## Domains
| Domain | Dataset | Metric |
|---|---|---|
| Medical | MedQA | Answer accuracy |
| Math | GSM8K | Exact numeric match |
| Coding | HumanEval | Token overlap score |
| Reasoning | HotpotQA | Answer inclusion |

## Tech stack
Python · FastAPI · LangGraph · Groq API · HuggingFace Datasets · SQLite · Render

## Run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API
- `POST /benchmark` — run one model on one domain
- `POST /compare` — compare multiple models side by side
- `GET /results` — all historical runs
- `GET /results/{domain}` — leaderboard by domain