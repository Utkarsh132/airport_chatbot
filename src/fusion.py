"""
Multimodal fusion / routing module.

Handles five input scenarios:
  1. text only
  2. image only
  3. voice only  (audio -> transcript -> routed as text)
  4. image + text
  5. voice + image (audio -> transcript, then treated as text + image)

Fusion strategy (documented per assignment "minimum requirement" tier):
  RULE-BASED ROUTING + WEIGHTED SIMILARITY MERGING.

  - Each available modality produces a candidate KB record + confidence score.
  - If text produced an exact entity match (e.g., gate number), that record's
    score is boosted (+0.3, capped at 1.0) because entity matches are highly
    reliable signals.
  - If image category and text/intent category agree, the shared record's
    score is boosted (+0.2, capped at 1.0) as cross-modal agreement increases
    trust in the result.
  - Final confidence below CONFIDENCE_LOW_THRESHOLD -> uncertain response,
    recommend human/information desk fallback.
  - Optional upgrade path (not required for minimum tier): replace the
    weighted merge with a small MLP taking [text_score, image_score,
    entity_match_flag, category_agreement_flag] as input, trained on labeled
    fusion outcomes.
"""

from config import CONFIDENCE_LOW_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD
from utils import normalize_score
from text_pipeline import process_query
from kb_retriever import KnowledgeBaseRetriever
from audio_pipeline import transcribe_audio

_INTENT_TO_CATEGORY = {
    "find_gate": "gate",
    "baggage_claim": "baggage_claim",
    "check_in": "check_in",
    "security": "security",
    "lounge": "lounge",
    "prayer_room": "lounge",
    "restaurant": "restaurant",
    "restroom": "restroom",
    "family_room": "lounge",
    "transport": "transport",
    "information_desk": "information_desk",
    "lost_and_found": "information_desk",
    "pharmacy": "restaurant",
    "atm": "information_desk",
    "immigration": "security",
    "car_rental": "transport",
    "medical_assistance": "information_desk",
    "special_assistance": "information_desk",
}


class MultimodalFusionEngine:
    def __init__(self, retriever: KnowledgeBaseRetriever = None):
        self.retriever = retriever or KnowledgeBaseRetriever()

    # ------------------------------------------------------------------
    def _text_branch(self, text: str):
        nlp = process_query(text)
        text_results = self.retriever.query_text(text)

        best = text_results[0] if text_results else None
        record = best["record"] if best else None
        score = normalize_score(best["score"]) if best else 0.0

        # entity exact-match boost (gate number)
        gate_entity = nlp["entities"].get("gate")
        if gate_entity:
            gate_record = self.retriever.kb_record_by_gate(gate_entity)
            if gate_record:
                record = gate_record
                score = min(1.0, score + 0.3)

        return {
            "nlp": nlp,
            "record": record,
            "score": score,
            "top_k": text_results,
        }

    def _image_branch(self, image_path_or_obj):
        image_results = self.retriever.query_image(image_path_or_obj)
        if not image_results:
            return {"category": None, "record": None, "score": 0.0, "top_k": []}

        best = image_results[0]
        category = best["category"]
        score = normalize_score(best["score"])

        kb_matches = self.retriever.kb_records_for_category(category)
        record = kb_matches[0] if kb_matches else None

        return {"category": category, "record": record, "score": score, "top_k": image_results}

    # ------------------------------------------------------------------
    def respond(self, text: str = None, image=None, audio_path: str = None):
        """
        Main entrypoint. Provide any combination of text, image, audio_path.
        Returns a structured response dict with: matched_record, confidence,
        rationale, uncertain (bool), and modality_breakdown.
        """
        transcript_info = None
        if audio_path:
            transcript_info = transcribe_audio(audio_path)
            if transcript_info["text"]:
                text = (text + " " + transcript_info["text"]).strip() if text else transcript_info["text"]

        text_branch = self._text_branch(text) if text else None
        image_branch = self._image_branch(image) if image is not None else None

        candidate_score = 0.0
        candidate_record = None
        rationale_parts = []

        if text_branch and text_branch["record"] is not None:
            candidate_record = text_branch["record"]
            candidate_score = text_branch["score"]
            rationale_parts.append(
                f"Text/voice query matched intent '{text_branch['nlp']['intent']}' "
                f"with retrieval score {candidate_score:.2f}."
            )

        if image_branch and image_branch["record"] is not None:
            if candidate_record is None:
                candidate_record = image_branch["record"]
                candidate_score = image_branch["score"]
                rationale_parts.append(
                    f"Image classified as category '{image_branch['category']}' "
                    f"with similarity score {candidate_score:.2f}."
                )
            else:
                same_category = candidate_record.get("category") == image_branch["category"]
                if same_category:
                    candidate_score = min(1.0, candidate_score + 0.2)
                    rationale_parts.append(
                        f"Image category '{image_branch['category']}' agrees with text intent -> "
                        f"confidence boosted to {candidate_score:.2f}."
                    )
                else:
                    rationale_parts.append(
                        f"Image category '{image_branch['category']}' conflicts with text-derived "
                        f"category '{candidate_record.get('category')}'; trusting higher-scoring modality."
                    )
                    if image_branch["score"] > candidate_score:
                        candidate_record = image_branch["record"]
                        candidate_score = image_branch["score"]

        uncertain = candidate_score < CONFIDENCE_LOW_THRESHOLD or candidate_record is None

        response = {
            "input_summary": {
                "text": text,
                "used_audio": audio_path is not None,
                "transcript": transcript_info["text"] if transcript_info else None,
                "transcript_engine": transcript_info["engine"] if transcript_info else None,
                "used_image": image is not None,
            },
            "matched_record": candidate_record,
            "confidence": round(candidate_score, 3),
            "uncertain": uncertain,
            "rationale": " ".join(rationale_parts) if rationale_parts else "No modality produced a confident match.",
            "modality_breakdown": {
                "text": text_branch,
                "image": image_branch,
            },
        }

        if uncertain:
            response["message"] = (
                "I'm not fully confident about this answer. Please visit the nearest "
                "Information Desk or check the official airport app/display screens "
                "for the most accurate and up-to-date information."
            )
        else:
            response["message"] = self._format_answer(candidate_record)

        return response

    @staticmethod
    def _format_answer(record: dict) -> str:
        if not record:
            return "No matching airport service found."
        return (
            f"{record['service_name']} ({record['category']}) is located at "
            f"{record['floor_or_zone']}, {record['terminal']}. "
            f"Directions: {record['directions']} "
            f"Opening hours: {record['opening_hours']}. "
            f"Accessibility: {record['accessibility']}"
        )


if __name__ == "__main__":
    engine = MultimodalFusionEngine()
    result = engine.respond(text="Where is gate B12?")
    print(result["message"])
    print("Confidence:", result["confidence"])
