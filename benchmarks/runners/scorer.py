import re

def score(model_answer: str, correct_answer: str, domain: str) -> dict:
    model_clean   = model_answer.strip().lower()
    correct_clean = correct_answer.strip().lower()

    if domain == "medical":
        is_correct = _score_medical(model_clean, correct_clean)
    elif domain == "math":
        is_correct = _score_math(model_clean, correct_clean)
    elif domain == "coding":
        is_correct = _score_coding(model_clean, correct_clean)
    elif domain == "reasoning":
        is_correct = _score_reasoning(model_clean, correct_clean)
    else:
        is_correct = model_clean == correct_clean

    is_hallucination = _detect_hallucination(model_answer, is_correct)

    return {"is_correct": is_correct, "is_hallucination": is_hallucination}

def _score_medical(model: str, correct: str) -> bool:
    match = re.search(r'\b([a-d])\b', model)
    if match:
        return match.group(1) == correct.lower()
    return correct.lower() in model

def _score_math(model: str, correct: str) -> bool:
    match = re.search(r'####\s*([\d,.-]+)', model)
    model_num = match.group(1).replace(",", "") if match else ""
    if not model_num:
        numbers = re.findall(r'[\d,.-]+', model)
        model_num = numbers[-1].replace(",", "") if numbers else ""
    return model_num == correct.replace(",", "")

def _score_coding(model: str, correct: str) -> bool:
    correct_tokens = set(re.findall(r'\b\w+\b', correct))
    model_tokens   = set(re.findall(r'\b\w+\b', model))
    if not correct_tokens:
        return False
    return len(correct_tokens & model_tokens) / len(correct_tokens) > 0.6

def _score_reasoning(model: str, correct: str) -> bool:
    return correct.lower() in model.lower()

def _detect_hallucination(model_answer: str, is_correct: bool) -> bool:
    uncertainty = ["i don't know", "i'm not sure", "i cannot", "unclear", "unknown"]
    admits_uncertainty = any(p in model_answer.lower() for p in uncertainty)
    return (not is_correct) and (not admits_uncertainty)