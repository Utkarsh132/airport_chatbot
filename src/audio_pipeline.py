"""
Audio preprocessing + speech-to-text pipeline.

Default (lightweight): faster-whisper "tiny" model.
Fallback (fully offline / no model download): a stub transcriber that
returns the ground-truth transcript from audio_metadata.json when available,
or an empty string otherwise -- this keeps the rest of the pipeline
demonstrable even without internet access.

Steps demonstrated: load audio -> (optional trimming) -> transcription ->
(optional) MFCC extraction for exploratory analysis.
"""

import os
import numpy as np

from config import USE_WHISPER, WHISPER_MODEL_SIZE, AUDIO_METADATA_PATH
from utils import load_json

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def load_audio(path: str, target_sr: int = 16000):
    """Load audio as a mono waveform at target_sr using librosa."""
    import librosa
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform, sr


def trim_silence(waveform: np.ndarray, top_db: int = 25) -> np.ndarray:
    """Trim leading/trailing silence using librosa's energy-based trimmer."""
    import librosa
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def extract_mfcc(waveform: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features -- useful for exploratory audio analysis / classification."""
    import librosa
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def _stub_transcribe(path: str) -> str:
    """Offline fallback: look up the ground-truth transcript by filename."""
    if os.path.exists(AUDIO_METADATA_PATH):
        meta = load_json(AUDIO_METADATA_PATH)
        filename = os.path.basename(path)
        for entry in meta:
            if entry["file"] == filename:
                return entry["transcript_ground_truth"]
    return ""


def transcribe_audio(path: str) -> dict:
    """
    Transcribe an audio file to text.
    Returns a dict: {"text": str, "engine": "whisper" | "stub", "avg_logprob": float or None}
    """
    if USE_WHISPER:
        try:
            model = _load_whisper()
            segments, info = model.transcribe(path, beam_size=1)
            text_parts = []
            logprobs = []
            for seg in segments:
                text_parts.append(seg.text)
                if seg.avg_logprob is not None:
                    logprobs.append(seg.avg_logprob)
            text = " ".join(text_parts).strip()
            avg_logprob = float(np.mean(logprobs)) if logprobs else None
            if text:
                return {"text": text, "engine": "whisper", "avg_logprob": avg_logprob}
        except Exception as e:
            print(f"[audio_pipeline] Whisper unavailable ({e}); using stub transcriber.")

    text = _stub_transcribe(path)
    return {"text": text, "engine": "stub", "avg_logprob": None}


COMMON_TRANSCRIPTION_ERROR_SOURCES = [
    "Background noise (announcements, crowd chatter) masking speech energy.",
    "Non-native accents shifting phoneme boundaries recognized by the acoustic model.",
    "Very short queries (2-4 words) giving the language model little context to disambiguate.",
    "Airport-specific terminology (gate codes like 'B12', airline names) not well represented "
    "in general-purpose training data.",
    "Overlapping speech or public address system interference.",
]


if __name__ == "__main__":
    print("Common transcription error sources documented for report:")
    for e in COMMON_TRANSCRIPTION_ERROR_SOURCES:
        print("-", e)
