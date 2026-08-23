"""
Text preprocessing, intent handling, entity extraction, and embedding
generation for the retrieval pipeline. Both typed queries and Whisper
transcripts are routed through this SAME pipeline, guaranteeing consistent
downstream behavior regardless of input modality.

Default (lightweight upgrade path): sentence-transformers (all-MiniLM-L6-v2)
for semantic embeddings.
Fallback (always available): scikit-learn TF-IDF vectorization.
"""

import re
import numpy as np

from config import USE_SENTENCE_TRANSFORMERS, SENTENCE_MODEL_NAME
from utils import clean_text

try:
    from rapidfuzz import fuzz, process as rf_process
    _RAPIDFUZZ_AVAILABLE = True
except Exception:
    _RAPIDFUZZ_AVAILABLE = False

_sentence_model = None
_tfidf_vectorizer = None

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "do", "does", "did", "i", "you", "my",
    "to", "of", "in", "on", "at", "for", "please", "can", "could", "there",
}

# Extra vocabulary for typo correction -- airport terms + common query words.
_EXTRA_VOCAB = {
    "where", "is", "how", "do", "i", "get", "to", "find", "near", "located",
    "location", "help", "please", "flight", "departure", "arrival", "terminal",
    "nearest", "closest", "directions", "open", "hours", "accessible",
    "wheelchair", "passport", "customs", "screening",
}


def _build_domain_vocabulary() -> set:
    """Words that directly drive intent matching -- corrected against FIRST
    so domain terms (gate, baggage, security...) win over generic words."""
    vocab = set()
    for keywords in INTENT_KEYWORDS.values():
        for kw in keywords:
            for tok in kw.split():
                vocab.add(tok)
    return vocab


def _build_vocabulary() -> set:
    return _build_domain_vocabulary() | set(_EXTRA_VOCAB)


def correct_typos(text: str, score_cutoff: int = 78) -> str:
    """
    Fuzzy spell-correction against a domain vocabulary using rapidfuzz.
    Domain-critical words (gate, baggage, security, etc.) are checked FIRST
    and with a slightly lower threshold, so misspellings like "gaet" resolve
    to "gate" rather than a generic word like "get". Only replaces a token
    when a close-enough match is found; otherwise the original token is kept
    (preserving gate codes, flight numbers, and truly unknown words).
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return text

    domain_vocab = _build_domain_vocabulary()
    full_vocab = _build_vocabulary()
    tokens = text.split()
    corrected_tokens = []
    for tok in tokens:
        if len(tok) < 3 or tok.isdigit() or tok in full_vocab:
            corrected_tokens.append(tok)
            continue

        domain_match = rf_process.extractOne(tok, domain_vocab, scorer=fuzz.ratio, score_cutoff=score_cutoff - 3)
        if domain_match:
            corrected_tokens.append(domain_match[0])
            continue

        generic_match = rf_process.extractOne(tok, full_vocab, scorer=fuzz.ratio, score_cutoff=score_cutoff)
        if generic_match:
            corrected_tokens.append(generic_match[0])
        else:
            corrected_tokens.append(tok)
    return " ".join(corrected_tokens)

GATE_RE = re.compile(r"\b([a-dA-D]\s?-?\s?\d{1,3})\b")
TERMINAL_RE = re.compile(r"\bterminal\s?(\d{1,2})\b", re.IGNORECASE)
FLIGHT_RE = re.compile(r"\b([A-Z]{2}\d{2,4})\b")
TIME_RE = re.compile(r"\b(\d{1,2}[:.]\d{2}\s?(am|pm)?)\b", re.IGNORECASE)

INTENT_KEYWORDS = {
    "find_gate": ["gate"],
    "baggage_claim": ["baggage", "luggage", "bag claim", "carousel"],
    "check_in": ["check in", "check-in", "checkin", "bag drop", "ticketing"],
    "security": ["security", "screening", "passport control line"],
    "lounge": ["lounge", "relax", "rest area"],
    "prayer_room": ["pray", "prayer"],
    "restaurant": ["eat", "food", "restaurant", "cafe", "hungry"],
    "restroom": ["restroom", "toilet", "wc", "bathroom"],
    "family_room": ["family room", "baby", "changing"],
    "transport": ["taxi", "train", "shuttle", "bus", "transport"],
    "information_desk": ["information", "info desk", "help point"],
    "lost_and_found": ["lost", "found", "missing item"],
    "pharmacy": ["pharmacy", "medicine", "medication"],
    "atm": ["atm", "cash", "currency", "money"],
    "immigration": ["passport control", "immigration", "border"],
    "car_rental": ["rent a car", "car rental"],
    "medical_assistance": ["medical", "doctor", "first aid", "injury"],
    "special_assistance": ["wheelchair", "special assistance", "disability", "mobility"],
    "flight_status": ["delayed", "delay", "on time", "flight status", "departure time"],
}


def tokenize(text: str) -> list:
    """Lowercase + clean + whitespace tokenization."""
    return clean_text(text).split()


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]


def extract_entities(text: str) -> dict:
    """Rule-based entity extraction for gate, terminal, flight number, and time."""
    entities = {}

    gate_match = GATE_RE.search(text)
    if gate_match:
        entities["gate"] = gate_match.group(1).upper().replace(" ", "").replace("-", "")

    terminal_match = TERMINAL_RE.search(text)
    if terminal_match:
        entities["terminal"] = f"Terminal {terminal_match.group(1)}"

    flight_match = FLIGHT_RE.search(text)
    if flight_match:
        entities["flight"] = flight_match.group(1)

    time_match = TIME_RE.search(text)
    if time_match:
        entities["time"] = time_match.group(1)

    return entities


def classify_intent_rule_based(text: str) -> str:
    """Keyword-based intent classifier with typo-tolerant fuzzy fallback."""
    lowered = clean_text(text)
    corrected = correct_typos(lowered)

    for candidate_text in [lowered, corrected]:
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in candidate_text:
                    return intent

    if _RAPIDFUZZ_AVAILABLE:
        best_intent = "unknown"
        best_score = 0
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                score = fuzz.partial_ratio(corrected, kw)
                if score > best_score:
                    best_score = score
                    best_intent = intent
        if best_score >= 80:
            return best_intent

    return "unknown"


def _load_sentence_model():
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    from sentence_transformers import SentenceTransformer
    _sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    return _sentence_model


def _fit_tfidf(corpus: list):
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf_vectorizer = TfidfVectorizer()
    _tfidf_vectorizer.fit(corpus)
    return _tfidf_vectorizer


def embed_text(text: str, corpus_for_tfidf: list = None) -> np.ndarray:
    """
    Generate an embedding vector for a text string.
    If USE_SENTENCE_TRANSFORMERS is True, use semantic embeddings.
    Otherwise fall back to TF-IDF (requires corpus_for_tfidf on first call).
    """
    if USE_SENTENCE_TRANSFORMERS:
        try:
            model = _load_sentence_model()
            vec = model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            print(f"[text_pipeline] sentence-transformers unavailable ({e}); using TF-IDF fallback.")

    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        if not corpus_for_tfidf:
            raise ValueError("TF-IDF fallback requires corpus_for_tfidf on first call.")
        _fit_tfidf(corpus_for_tfidf)
    vec = _tfidf_vectorizer.transform([text]).toarray()[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def process_query(text: str) -> dict:
    """
    Unified text-processing entrypoint used for BOTH typed queries and
    Whisper transcripts, guaranteeing identical downstream behavior.
    """
    cleaned_text = clean_text(text)
    corrected_text = correct_typos(cleaned_text)
    tokens = tokenize(corrected_text)
    tokens_no_stop = remove_stopwords(tokens)
    intent = classify_intent_rule_based(corrected_text)
    entities = extract_entities(text)
    return {
        "raw_text": text,
        "cleaned_text": cleaned_text,
        "corrected_text": corrected_text,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stop,
        "intent": intent,
        "entities": entities,
    }


if __name__ == "__main__":
    sample = "Where is gate B12? Is flight LH123 delayed?"
    result = process_query(sample)
    print(result)
