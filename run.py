"""
End-to-end command-line demo runner.

Usage:
    python run.py --build-data      # generate synthetic images (first run only)
    python run.py --evaluate        # run full evaluation suite, save outputs/
    python run.py --query "Where is gate B12?"
    python run.py --image data/images/gate/gate_00.png --query "Where is this?"
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="Airport Multimodal Chatbot demo runner")
    parser.add_argument("--build-data", action="store_true", help="Generate synthetic image dataset")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation suite")
    parser.add_argument("--query", type=str, default=None, help="Text query to ask the chatbot")
    parser.add_argument("--image", type=str, default=None, help="Path to an image to ask about")
    parser.add_argument("--audio", type=str, default=None, help="Path to an audio file to transcribe and ask")
    args = parser.parse_args()

    if args.build_data:
        from dataset_builder import build_dataset
        manifest = build_dataset(images_per_category=8)
        print(f"Generated {len(manifest)} synthetic images.")
        return

    if args.evaluate:
        from kb_retriever import KnowledgeBaseRetriever
        from fusion import MultimodalFusionEngine
        from evaluation import evaluate_vision, evaluate_speech, evaluate_text_retrieval, evaluate_fusion_scenarios

        retriever = KnowledgeBaseRetriever()
        print("=== Vision Evaluation ===")
        print(evaluate_vision(retriever))
        print("=== Speech Evaluation ===")
        print(evaluate_speech())
        print("=== Text/Retrieval Evaluation ===")
        print(evaluate_text_retrieval(retriever))

        engine = MultimodalFusionEngine(retriever)
        print("=== Fusion Scenarios ===")
        for r in evaluate_fusion_scenarios(engine):
            print(r)
        return

    if args.query or args.image or args.audio:
        from fusion import MultimodalFusionEngine
        from PIL import Image

        engine = MultimodalFusionEngine()
        image_obj = Image.open(args.image).convert("RGB") if args.image else None
        result = engine.respond(text=args.query, image=image_obj, audio_path=args.audio)

        print("Message:", result["message"])
        print("Confidence:", result["confidence"])
        print("Uncertain:", result["uncertain"])
        print("Rationale:", result["rationale"])
        return

    parser.print_help()


if __name__ == "__main__":
    main()
