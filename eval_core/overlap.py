import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "it", "this", "that", "with", "i", "my", "me", "we",
    "you", "she", "her", "he", "his", "what", "should", "do", "does", "can",
    "will", "if", "how", "when", "tell", "say", "patient", "didi", "about",
    "not", "have", "has",
}


def content_tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


def overlap_score(a, b):
    """Share of the smaller word set found in the other. 0 if either is too short."""
    ta, tb = content_tokens(a), content_tokens(b)
    if len(ta) < 3 or len(tb) < 3:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def max_overlap(text, others):
    return max((overlap_score(text, other) for other in others), default=0.0)


def best_match(text, others):
    """Return (score, index) of the closest other text. (0.0, -1) if others is empty."""
    best_score, best_index = 0.0, -1
    for index, other in enumerate(others):
        score = overlap_score(text, other)
        if score > best_score:
            best_score, best_index = score, index
    return best_score, best_index
