"""
Global configuration for the Airport Multimodal Chatbot proof-of-concept.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TEXT_DIR = os.path.join(DATA_DIR, "text")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

KB_JSON_PATH = os.path.join(KB_DIR, "airport_kb.json")
KB_CSV_PATH = os.path.join(KB_DIR, "airport_kb.csv")
TEXT_DATASET_PATH = os.path.join(TEXT_DIR, "passenger_queries.json")
AUDIO_METADATA_PATH = os.path.join(AUDIO_DIR, "audio_metadata.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

IMAGE_CATEGORIES = [
    "gate", "baggage_claim", "check_in", "security", "restroom",
    "lounge", "transport", "information_desk", "restaurant",
]

USE_CLIP = True
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224

USE_WHISPER = True
WHISPER_MODEL_SIZE = "tiny"

USE_SENTENCE_TRANSFORMERS = False
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"

CONFIDENCE_LOW_THRESHOLD = 0.20
CONFIDENCE_HIGH_THRESHOLD = 0.65
TOP_K_RETRIEVAL = 3

RANDOM_SEED = 42
