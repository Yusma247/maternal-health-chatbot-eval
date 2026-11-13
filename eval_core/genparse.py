import json
import re


def parse_qa_json(text):
    """Turn a model reply into a list of {question, answer}. Bad output gives []."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    pairs = []
    for item in data:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            pairs.append({
                "question": str(item["question"]).strip(),
                "answer": str(item["answer"]).strip(),
            })
    return pairs
