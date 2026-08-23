"""
Basic smoke tests for the Airport Multimodal Chatbot pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from text_pipeline import process_query, extract_entities, classify_intent_rule_based  # noqa: E402
from utils import simple_word_error_rate, cosine_similarity, clean_text  # noqa: E402
from data_loader import load_knowledge_base, load_text_dataset  # noqa: E402


def test_knowledge_base_has_minimum_records():
    kb = load_knowledge_base()
    assert len(kb) >= 15, "Knowledge base must have at least 15-20 records"


def test_knowledge_base_required_fields():
    kb = load_knowledge_base()
    required = {"id", "service_name", "category", "terminal", "floor_or_zone",
                "short_description", "opening_hours", "directions",
                "accessibility", "related_facilities", "emergency_or_help_contact"}
    assert required.issubset(set(kb.columns))


def test_text_dataset_loads():
    df = load_text_dataset()
    assert len(df) > 0
    assert "intent" in df.columns


def test_entity_extraction_gate():
    entities = extract_entities("Where is gate B12?")
    assert entities.get("gate") == "B12"


def test_entity_extraction_terminal():
    entities = extract_entities("Is there a lounge near terminal 2?")
    assert entities.get("terminal") == "Terminal 2"


def test_intent_classification_basic():
    assert classify_intent_rule_based("Where is gate B12?") == "find_gate"
    assert classify_intent_rule_based("How do I get to baggage claim?") == "baggage_claim"


def test_process_query_pipeline():
    result = process_query("Where Is GATE b12??")
    assert result["cleaned_text"] == "where is gate b12"
    assert "gate" in result["tokens"]


def test_word_error_rate_identical():
    assert simple_word_error_rate("hello world", "hello world") == 0.0


def test_word_error_rate_completely_different():
    wer = simple_word_error_rate("hello world", "goodbye moon")
    assert wer == 1.0


def test_cosine_similarity_identical_vectors():
    import numpy as np
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_clean_text():
    assert clean_text("Where is GATE B12?!") == "where is gate b12"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
