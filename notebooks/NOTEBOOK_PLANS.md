# Notebook Plans

## 01_data_exploration.ipynb
1. Load knowledge base (`data_loader.load_knowledge_base`) and display as DataFrame.
2. Load text dataset and display sample queries with intent labels.
3. Build image manifest and display representative sample images per category (grid plot).
4. Plot class distribution (bar chart) for image categories and text intents.
5. Compare similar vs dissimilar image examples (e.g., two "gate" images vs one "gate" + one "security").
6. Discuss dataset limitations: synthetic images, small sample size, no lighting/perspective variation,
   generalization risk to real-world signage.
7. Vocabulary analysis: token frequency, average query length, word cloud (optional).
8. Entity examples: print extracted gate/terminal/flight/time entities for 5-10 sample queries.

## 02_preprocessing.ipynb
1. Image pipeline demo: load -> resize -> normalize -> augment -> tensor batch, with before/after visualization.
2. Audio pipeline demo: load metadata, explain trimming/MFCC extraction (librosa), show sample MFCC heatmap.
3. Text pipeline demo: tokenization, stopword removal, entity extraction, before/after examples table.
4. Train/validation split of text dataset, display split sizes.
5. Embedding generation demo: show embedding vector shape and a 2D PCA/TSNE plot of query embeddings colored by intent.

## 03_model_pipeline.ipynb
1. Initialize `KnowledgeBaseRetriever`, show text index build time and vector shapes.
2. Run sample text queries through retrieval, display top-3 KB matches with scores.
3. Run sample image queries through retrieval, display top-3 category matches with scores.
4. Demonstrate `MultimodalFusionEngine.respond()` for all 5 scenarios (text-only, image-only,
   voice-only, image+text, voice+image) with printed rationale and confidence.
5. Show a deliberately low-confidence/ambiguous query and confirm the uncertainty fallback message.

## 04_evaluation.ipynb
1. Run `evaluate_vision()`, display top-1/top-3 accuracy bar chart and examples table.
2. Run `evaluate_speech()`, display transcript comparison table and mean WER.
3. Run `evaluate_text_retrieval()`, display precision/recall/F1 and confusion matrix heatmap.
4. Run `evaluate_fusion_scenarios()`, display scenario comparison table.
5. Discussion cells: performance gaps, failure cases, dataset limitations, suggested improvements.
