# LLM Benchmark Engine

Measures LLM accuracy, hallucination rate, and latency across four domains using real public datasets.

## Live Demo
- API: https://llm-benchmark-engine.onrender.com
- Docs: https://llm-benchmark-engine.onrender.com/docs
- Results: https://llm-benchmark-engine.onrender.com/results

## Models tested
| Model | Parameters | Provider |
|---|---|---|
| Llama 3.3 70B | 70B | Groq |
| Llama 3.1 8B | 8B | Groq |
| Qwen3 32B | 32B | Groq |

## Benchmark datasets
| Domain | Dataset | Metric |
|---|---|---|
| Math | GSM8K (openai/gsm8k) | Exact numeric match |
| Medical | MedMCQA (openlifescienceai/medmcqa) | Multiple choice accuracy |
| Coding | HumanEval (openai/openai_humaneval) | Token overlap score |
| Reasoning | TriviaQA (lucadiliello/triviaqa) | Answer inclusion |

## Key findings (240 samples · 20 per run)
| Model | Math | Medical | Coding | Reasoning |
|---|---|---|---|---|
| llama3-70b | 95% | 70% | 70% | 80% |
| llama3-8b  | 80% | 50% | 60% | 70% |
| qwen-32b   | 35% | 25% | 65% | 85% |

- llama3-70b achieves 95% accuracy on GSM8K with only 5% hallucination rate
- qwen-32b scores 85% on reasoning but 35% on math — thinking-token format mismatch with structured output scorer
- llama3-70b is faster than llama3-8b on math (669ms vs 2369ms) — Groq LPU hardware advantage
- qwen-32b coding latency 5.9s vs 299ms for llama3-70b — thinking tokens inflate response time 20x

## Tech stack
Python · FastAPI · Groq API · HuggingFace Datasets · SQLite · Render

## Run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API endpoints
- `POST /benchmark` — run one model on one domain
- `POST /compare` — compare multiple models side by side
- `GET /results` — all historical runs
- `GET /results/{domain}` — leaderboard by domain

## Note on live data
The free Render tier does not persist disk storage between deploys.
To populate results after waking the service, run any `/benchmark` request via the Swagger UI.
In production this would use PostgreSQL — the ORM is already SQLAlchemy so it is a one-line connection string change.