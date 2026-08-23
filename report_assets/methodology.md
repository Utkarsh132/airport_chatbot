# Methodology

## Environment Setup
The system is implemented in Python 3.11 using a modular pipeline architecture. Core dependencies
include PyTorch and Transformers (vision), faster-whisper and Librosa (speech), scikit-learn and
sentence-transformers (text), FAISS (similarity search), and Streamlit (deployment). Every heavy
model has a lightweight, dependency-free fallback (TF-IDF for text, color-histogram descriptors
for images, a ground-truth stub for speech), so the system remains runnable on a standard student
laptop without GPU access or reliable internet connectivity.

## Data Acquisition and Exploration
Because licensed real-world airport imagery could not be sourced without copyright and privacy
concerns, a synthetic image dataset was generated programmatically: 8 sign-style images per
category across 9 categories (72 images total), each rendering category-appropriate iconography
and text labels (e.g., "GATE B12", "BAGGAGE CLAIM"). This approach avoids licensing issues while
producing images with enough visual structure for CLIP-style embeddings to discriminate between
categories. A companion 30-query text dataset with intent labels and entity annotations was
manually curated to reflect realistic passenger questions. A 5-sample audio metadata file
documents expected transcripts and recording conditions (accent, noise, query length) to support
speech evaluation without requiring bundled audio recordings.

## Preprocessing
Images are loaded, resized to 224x224, optionally augmented (rotation ±8°, brightness jitter),
normalized to [0,1], and converted to tensors/batches for embedding. Audio is loaded at 16kHz mono,
optionally trimmed of silence, and transcribed via Whisper (or MFCC-extracted for exploratory
analysis). Text queries -- whether typed or transcribed from voice -- are routed through an
identical NLP pipeline: lowercasing, tokenization, stopword removal, rule-based intent
classification, and regex-based entity extraction (gate, terminal, flight number, time), ensuring
consistent behavior regardless of input modality.

## Model Design
The vision module uses CLIP embeddings indexed with FAISS for image-to-category retrieval. The
speech module uses Whisper (tiny, CPU, int8 quantized) for transcription. The text module uses
sentence-transformers for semantic similarity against knowledge base descriptions, boosted by
exact entity matches (e.g., a recognized gate number directly resolves to that gate's record).
The fusion module implements rule-based routing: each modality proposes a candidate KB record and
confidence; cross-modal category agreement boosts confidence, disagreement defers to the
higher-scoring modality, and low overall confidence triggers an explicit uncertainty response
recommending human/staff fallback.

## Training and Evaluation
All models are used frozen (pretrained); no fine-tuning was performed given dataset size
constraints, consistent with the assignment's frozen-model evaluation approach. Evaluation covers
vision (top-1/top-3 retrieval accuracy), speech (Word Error Rate against ground-truth transcripts),
text (retrieval accuracy, precision/recall/F1 on intent labels, confusion matrix), and multimodal
fusion (structured comparison across the five input scenarios).

## Deployment
A Streamlit prototype provides text input, audio upload, image upload, a response panel, a
confidence indicator, and explicit uncertainty warnings, along with a persistent privacy notice
discouraging upload of sensitive documents.
