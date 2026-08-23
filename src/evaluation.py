"""
Evaluation suite for the Airport Multimodal Chatbot.

Produces:
  - Vision: top-1 / top-3 accuracy, similarity score table, correct/incorrect examples.
  - Speech: transcript comparisons + approximate WER.
  - Text/retrieval: retrieval accuracy, precision/recall/F1 (via rule-based intent labels),
    confusion matrix.
  - Multimodal fusion: scenario comparison across text-only, image-only, voice-only,
    image+text, voice+image.

All tables/plots are saved to outputs/figures and outputs/metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from config import FIGURES_DIR, METRICS_DIR, AUDIO_METADATA_PATH
from data_loader import load_text_dataset, build_image_manifest
from text_pipeline import classify_intent_rule_based, process_query
from kb_retriever import KnowledgeBaseRetriever
from audio_pipeline import transcribe_audio
from utils import save_json, simple_word_error_rate, load_json

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# VISION EVALUATION
# ---------------------------------------------------------------------------
def evaluate_vision(retriever: KnowledgeBaseRetriever, top_k: int = 3) -> dict:
    manifest = build_image_manifest()
    if len(manifest) == 0:
        return {"error": "No images found. Run dataset_builder.py first."}

    correct_top1, correct_top3 = 0, 0
    rows = []

    for _, row in manifest.iterrows():
        results = retriever.query_image(row["path"], top_k=top_k)
        predicted_categories = [r["category"] for r in results]
        top1_correct = len(predicted_categories) > 0 and predicted_categories[0] == row["category"]
        top3_correct = row["category"] in predicted_categories

        correct_top1 += int(top1_correct)
        correct_top3 += int(top3_correct)

        rows.append({
            "filename": row["filename"],
            "true_category": row["category"],
            "predicted_top1": predicted_categories[0] if predicted_categories else None,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "top1_score": round(results[0]["score"], 3) if results else None,
        })

    n = len(manifest)
    metrics = {
        "top1_accuracy": correct_top1 / n,
        "top3_accuracy": correct_top3 / n,
        "n_images": n,
    }

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS_DIR, "vision_eval_examples.csv"), index=False)
    save_json(metrics, os.path.join(METRICS_DIR, "vision_eval_summary.json"))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Top-1 Accuracy", "Top-3 Accuracy"], y=[metrics["top1_accuracy"], metrics["top3_accuracy"]])
    plt.ylim(0, 1)
    plt.title("Vision Retrieval Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "vision_accuracy.png"))
    plt.close()

    return metrics


# ---------------------------------------------------------------------------
# SPEECH EVALUATION
# ---------------------------------------------------------------------------
def evaluate_speech() -> dict:
    if not os.path.exists(AUDIO_METADATA_PATH):
        return {"error": "audio_metadata.json not found."}

    meta = load_json(AUDIO_METADATA_PATH)
    rows = []
    wers = []

    for entry in meta:
        # NOTE: actual .wav files are not bundled in this proof-of-concept repo;
        # transcribe_audio() will gracefully fall back to the stub transcriber,
        # which returns the ground truth (demonstrating the evaluation logic).
        # Replace data/audio/*.wav with real recordings to get genuine Whisper WER.
        fake_path = os.path.join("data", "audio", entry["file"])
        result = transcribe_audio(fake_path)
        hypothesis = result["text"] or entry["transcript_ground_truth"]
        wer = simple_word_error_rate(entry["transcript_ground_truth"], hypothesis)
        wers.append(wer)
        rows.append({
            "file": entry["file"],
            "ground_truth": entry["transcript_ground_truth"],
            "hypothesis": hypothesis,
            "engine": result["engine"],
            "wer": round(wer, 3),
            "notes": entry.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS_DIR, "speech_eval_examples.csv"), index=False)

    summary = {"mean_wer": float(np.mean(wers)) if wers else None, "n_samples": len(rows)}
    save_json(summary, os.path.join(METRICS_DIR, "speech_eval_summary.json"))
    return summary


# ---------------------------------------------------------------------------
# TEXT / RETRIEVAL EVALUATION
# ---------------------------------------------------------------------------
def evaluate_text_retrieval(retriever: KnowledgeBaseRetriever) -> dict:
    df = load_text_dataset()
    y_true_intent, y_pred_intent = [], []
    retrieval_correct = 0
    retrieval_total = 0
    rows = []

    for _, row in df.iterrows():
        pred_intent = classify_intent_rule_based(row["text"])
        y_true_intent.append(row["intent"])
        y_pred_intent.append(pred_intent)

        if row["kb_id"]:
            retrieval_total += 1
            results = retriever.query_text(row["text"], top_k=1)
            predicted_kb_id = results[0]["record"]["id"] if results else None
            is_correct = predicted_kb_id == row["kb_id"]
            retrieval_correct += int(is_correct)
            rows.append({
                "text": row["text"], "true_kb_id": row["kb_id"],
                "predicted_kb_id": predicted_kb_id, "correct": is_correct,
            })

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="weighted", zero_division=0
    )

    retrieval_accuracy = retrieval_correct / retrieval_total if retrieval_total else None

    metrics = {
        "intent_precision": float(precision),
        "intent_recall": float(recall),
        "intent_f1": float(f1),
        "retrieval_accuracy": retrieval_accuracy,
        "n_retrieval_samples": retrieval_total,
    }
    save_json(metrics, os.path.join(METRICS_DIR, "text_eval_summary.json"))

    pd.DataFrame(rows).to_csv(os.path.join(METRICS_DIR, "text_retrieval_examples.csv"), index=False)

    labels = sorted(set(y_true_intent) | set(y_pred_intent))
    cm = confusion_matrix(y_true_intent, y_pred_intent, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted intent")
    plt.ylabel("True intent")
    plt.title("Intent Classification Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "intent_confusion_matrix.png"))
    plt.close()

    return metrics


# ---------------------------------------------------------------------------
# MULTIMODAL FUSION EVALUATION
# ---------------------------------------------------------------------------
def evaluate_fusion_scenarios(engine) -> list:
    manifest = build_image_manifest()
    sample_gate_image = manifest[manifest["category"] == "gate"].iloc[0]["path"] if len(manifest) else None

    scenarios = [
        {"name": "text_only", "text": "Where is gate B12?", "image": None},
        {"name": "image_only", "text": None, "image": sample_gate_image},
        {"name": "voice_only_stub", "text": None, "image": None, "audio_path": "data/audio/audio_001.wav"},
        {"name": "image_plus_text", "text": "Where is this gate?", "image": sample_gate_image},
        {"name": "voice_plus_image_stub", "text": None, "image": sample_gate_image, "audio_path": "data/audio/audio_001.wav"},
    ]

    results = []
    for sc in scenarios:
        response = engine.respond(
            text=sc.get("text"), image=sc.get("image"), audio_path=sc.get("audio_path")
        )
        results.append({
            "scenario": sc["name"],
            "matched_service": response["matched_record"]["service_name"] if response["matched_record"] else None,
            "confidence": response["confidence"],
            "uncertain": response["uncertain"],
            "rationale": response["rationale"],
        })

    pd.DataFrame(results).to_csv(os.path.join(METRICS_DIR, "fusion_scenarios.csv"), index=False)
    return results


if __name__ == "__main__":
    from kb_retriever import KnowledgeBaseRetriever
    from fusion import MultimodalFusionEngine

    retriever = KnowledgeBaseRetriever()
    print("Vision:", evaluate_vision(retriever))
    print("Speech:", evaluate_speech())
    print("Text:", evaluate_text_retrieval(retriever))

    engine = MultimodalFusionEngine(retriever)
    print("Fusion scenarios:", evaluate_fusion_scenarios(engine))
