#!/bin/bash

MODELS=("llama3-70b" "llama3-8b" "qwen")
DOMAINS=("math" "reasoning" "medical" "coding")

for domain in "${DOMAINS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running $model on $domain..."
    curl -s -X POST "http://127.0.0.1:8000/benchmark" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$model\", \"domain\": \"$domain\", \"num_samples\": 20}" \
      | python3 -m json.tool
    echo ""
    sleep 2
  done
done

echo "All benchmarks complete."