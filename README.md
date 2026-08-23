# Airport Multimodal Passenger Assistance Chatbot

Proof-of-concept multimodal chatbot for airport passenger assistance (image, voice, text input).
Built for MSc AI coursework demonstrating the full multimodal AI pipeline: data acquisition,
preprocessing, model design, multimodal fusion, evaluation, deployment, and ethics.

## Architecture

- **Vision**: CLIP (openai/clip-vit-base-patch32) embeddings + FAISS retrieval, with an offline
  color-histogram fallback if torch/transformers are unavailable.
- **Speech**: faster-whisper ("tiny" model by default), with a stub transcriber fallback for
  fully offline demos.
- **Text**: sentence-transformers (all-MiniLM-L6-v2) semantic retrieval + rule-based intent
  classification and entity extraction, with a TF-IDF fallback.
- **Fusion**: rule-based routing + weighted confidence merging across modalities (text, image,
  voice, image+text, voice+image).
- **Knowledge base**: 20 structured airport service/location records (JSON + CSV).
- **UI**: Streamlit prototype with text/image/audio upload and confidence display.

All heavy models are optional — set flags in `src/config.py` (`USE_CLIP`, `USE_WHISPER`,
`USE_SENTENCE_TRANSFORMERS`) to `False` to run entirely offline with lightweight fallbacks.
This was verified: the fallback-mode pipeline was tested end-to-end and correctly resolves both
text and image queries to the right knowledge base record.

## Project Structure

```
airport_multimodal_chatbot/
├── app/                    Streamlit UI + inference wrapper
├── data/                   Images, audio metadata, text dataset, knowledge base
├── src/                    Core pipeline modules (config, pipelines, fusion, evaluation)
├── notebooks/              Notebook plans (see NOTEBOOK_PLANS.md)
├── outputs/                Generated figures, metrics, predictions
├── tests/                  Pytest smoke tests
├── report_assets/          Report-ready writeup material
├── requirements.txt
├── Dockerfile
└── run_demo.py             CLI entry point
```

## Setup

Recommended Python version: 3.10 or 3.11 (3.12 also works, tested).

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Lightweight mode (no heavy downloads, runs on any laptop)

Edit `src/config.py` and set:
```python
USE_CLIP = False
USE_WHISPER = False
USE_SENTENCE_TRANSFORMERS = False
```
This uses TF-IDF for text retrieval and a color-histogram descriptor for images. No GPU,
no internet download required after `pip install`.

### Full-quality mode (recommended if you have time/bandwidth)

Leave the flags as `True` (default) to use CLIP, Whisper, and sentence-transformers.
First run will download model weights (~500MB-1.5GB total).

## How to Run

1. Generate the synthetic image dataset (first run only):
   ```bash
   python run_demo.py --build-data
   ```

2. Ask a question via CLI:
   ```bash
   python run_demo.py --query "Where is gate B12?"
   python run_demo.py --image data/images/gate/gate_00.png
   ```

3. Run the full evaluation suite (saves tables/plots to outputs/):
   ```bash
   python run_demo.py --evaluate
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. Run tests:
   ```bash
   pytest tests/test_pipeline.py -v
   ```

## Docker (optional)

```bash
docker build -t airport-chatbot .
docker run -p 8501:8501 airport-chatbot
```
Then open http://localhost:8501

## Known Limitations

- Image dataset is synthetic (generated sign-style cards), not real photographs — documented
  as a dataset limitation per assignment requirements. Replace `data/images/<category>/` with
  real or curated photos for higher realism.
- No real .wav files are bundled; `data/audio/audio_metadata.json` provides ground-truth
  transcripts used by the stub transcriber so the evaluation pipeline is demonstrable without
  audio files. Add real recordings to `data/audio/` for genuine Whisper WER results.
- Flight status queries (e.g., "Is my flight delayed?") are intentionally NOT resolved from the
  static knowledge base, since real-time flight data requires a live airline/airport API —
  the chatbot correctly returns an uncertainty message and redirects to official sources.

## Ethics & Privacy

See `report_assets/ethics.md` for the full discussion tied to this system's design (GDPR,
data minimisation, accent bias, accessibility, human handover).
