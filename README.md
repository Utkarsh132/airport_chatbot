#  Airport Multimodal Passenger Assistance Chatbot

Proof-of-concept multimodal chatbot for airport passenger assistance (text, voice, and image input).
Built for MSc AI coursework demonstrating the full multimodal AI pipeline: data acquisition,
preprocessing, model design, multimodal fusion, evaluation, deployment, and ethics.

---

## Architecture

| Modality | Model | Fallback (offline) |
|---|---|---|
| Text | sentence-transformers `all-MiniLM-L6-v2` + rule-based intent classifier | TF-IDF cosine similarity |
| Vision | CLIP `openai/clip-vit-base-patch32` + FAISS retrieval | Colour-histogram descriptor |
| Speech | faster-whisper (`tiny` model) | Ground-truth transcript lookup |
| Fusion | Weighted confidence merging across all active modalities | Same logic, lower scores |

**Knowledge base:** 20 structured airport service/location records (`data/knowledge_base/`).

**UI:** Streamlit chat interface — single input bar with text, image upload, audio upload, camera, and live mic.

---

## Project Structure

```
airport_multimodal_chatbot/
├── app/
│   ├── streamlit_app.py      # Streamlit UI
│   ├── inference.py          # Inference wrapper (text/image/audio → result)
│   └── ui_helpers.py         # UI formatting helpers
├── data/
│   ├── audio/                # 16 synthetic .wav files + audio_metadata.json
│   ├── images/               # Synthetic sign images per KB category
│   ├── knowledge_base/       # airport_kb.json + airport_kb.csv
│   └── text/                 # passenger_queries.json (intent training data)
├── src/
│   ├── config.py             # All flags and constants (USE_CLIP, USE_WHISPER, etc.)
│   ├── text_pipeline.py      # Intent classification + KB retrieval
│   ├── image_pipeline.py     # CLIP / histogram image retrieval
│   ├── audio_pipeline.py     # Whisper transcription
│   ├── fusion.py             # Multimodal fusion engine
│   ├── kb_retriever.py       # Knowledge base loader
│   ├── evaluation.py         # Evaluation metrics
│   ├── dataset_builder.py    # Synthetic data generation
│   ├── data_loader.py        # Data loading utilities
│   └── utils.py              # Shared helpers
├── tests/
│   └── test_pipeline.py      # 15 pytest smoke tests (all passing)
├── outputs/
│   ├── figures/              # Evaluation charts (confusion matrix, accuracy)
│   └── metrics/              # Evaluation JSON/CSV results
├── report_assets/            # Report writeup material (methodology, ethics, tables)
├── requirements.txt
├── run.py                    # CLI entry point
└── README.md
```

---

## ⚡ Quick Setup (Recommended: conda)

> Tested on Python 3.11, macOS (Apple Silicon) and Linux.
> A conda environment is strongly recommended to avoid dependency conflicts with
> `torch`, `faiss`, and `faster-whisper`.

### Step 1 — Create the conda environment

```bash
conda create -n airport-chatbot python=3.11 -y
conda activate airport-chatbot
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download model weights for CLIP (~350 MB), sentence-transformers (~90 MB),
> and faster-whisper tiny (~75 MB). Requires internet on first run only; cached after that.

### Step 3 — Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ Alternative: pip + venv (no conda)

If you prefer a plain virtual environment:

```bash
python3.11 -m venv .env
source .env/bin/activate        # Windows: .env\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> **Note for Apple Silicon (M1/M2/M3):** if `torch` or `faiss-cpu` fail to install,
> use conda (above) as it resolves Apple Silicon binaries automatically.

---

## 🔌 Offline / No-Internet Mode

If there is no internet access, the app still runs using lightweight fallbacks.
Edit `src/config.py` and set:

```python
USE_CLIP = False
USE_WHISPER = False
USE_SENTENCE_TRANSFORMERS = False
```

This uses TF-IDF for text retrieval and a colour-histogram descriptor for images.
No model downloads required. All 15 tests pass in this mode.

Alternatively, set the environment variable before running:

```bash
TRANSFORMERS_OFFLINE=1 streamlit run app/streamlit_app.py
```

---

## 🧪 Running Tests

```bash
conda activate airport-chatbot
cd airport_multimodal_chatbot
pytest tests/test_pipeline.py -v
```

Expected result: **15/15 passed**.

---

## 🖱️ CLI Usage

```bash
# Ask a text question
python run.py --query "Where is gate B12?"

# Query with an image
python run.py --image data/images/restroom/restroom_00.png

# Build/regenerate the synthetic image dataset
python run.py --build-data

# Run the full evaluation suite (saves results to outputs/)
python run.py --evaluate
```

---

## Known Limitations

- **Image dataset is synthetic** — generated sign-style icons, not real photographs.
  Documented as a dataset limitation. Replace `data/images/<category>/` with real photos
  for higher realism.
- **Flight status queries** are intentionally not resolved from the static knowledge base
  (e.g. "Is my flight delayed?"). The chatbot correctly returns an uncertainty message and
  redirects to official sources, since real-time data requires a live airline API.
- **Whisper on TTS-generated audio** occasionally mis-transcribes phonetically similar words
  (e.g. "restroom" → "restaurant"). Real human speech performs more reliably.
- **Confidence scores in offline mode** are lower than full-model mode (TF-IDF vs.
  sentence-transformers) — the low-confidence threshold is calibrated accordingly in
  `src/config.py`.

---

## Ethics & Privacy

See `report_assets/ethics.md` for the full discussion: GDPR compliance, data minimisation,
accent/language bias, accessibility, and human handover design.
