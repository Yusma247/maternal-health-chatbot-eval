import re

NEGATION_CUES = (
    "avoid",
    "do not",
    "don't",
    "dont",
    "never",
    "not safe",
    "should not",
    "shouldn't",
    "not recommended",
    "not advised",
    "cannot",
    "can't",
    "must not",
)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip().lower() for p in parts if p.strip()]


def check_required(response, groups):
    """groups is a list of lists. A group passes if any phrase in it appears."""
    text = response.lower()
    missing = [g for g in groups if not any(p.lower() in text for p in g)]
    return {"passed": len(groups) - len(missing), "total": len(groups), "missing": missing}


def check_forbidden(response, phrases):
    """A forbidden phrase counts only in a sentence with no negation cue."""
    violations = []
    for sentence in split_sentences(response):
        if any(cue in sentence for cue in NEGATION_CUES):
            continue
        for phrase in phrases:
            if phrase.lower() in sentence:
                violations.append({"phrase": phrase, "sentence": sentence})
    return {"violations": violations}
