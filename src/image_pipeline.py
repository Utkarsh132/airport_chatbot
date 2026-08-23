"""
Image preprocessing + vision embedding pipeline.

Default (lightweight): CLIP (openai/clip-vit-base-patch32) via transformers,
with FAISS for nearest-neighbor retrieval against the image manifest / KB
descriptions.

Fallback (fully offline, no downloads): a simple color-histogram + edge
descriptor. This keeps the pipeline runnable even without internet access
or a transformers/torch installation, at the cost of accuracy.

Preprocessing steps demonstrated: load -> resize -> normalize -> (optional
augmentation) -> tensor conversion -> batch embedding.
"""

import os
import numpy as np
from PIL import Image, ImageEnhance

from config import IMAGE_SIZE, USE_CLIP, CLIP_MODEL_NAME

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    from transformers import CLIPModel, CLIPProcessor
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _clip_model.eval()
    return _clip_model, _clip_processor


def load_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    return Image.open(path).convert("RGB")


def resize_image(img: Image.Image, size=IMAGE_SIZE) -> Image.Image:
    return img.resize((size, size))


def augment_image(img: Image.Image, seed: int = None) -> Image.Image:
    """Light augmentation: random brightness + slight rotation, for robustness demos."""
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-8, 8))
    brightness_factor = float(rng.uniform(0.85, 1.15))

    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    return img


def normalize_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to a normalized float32 numpy array in [0, 1], HWC."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_tensor_batch(images: list) -> np.ndarray:
    """Stack a list of normalized HWC arrays into an NHWC batch."""
    return np.stack([normalize_array(resize_image(img)) for img in images], axis=0)


def _fallback_descriptor(img: Image.Image) -> np.ndarray:
    """
    Offline fallback embedding: normalized RGB color histogram (32 bins/channel)
    concatenated with a crude edge-density scalar. Not as semantically rich as
    CLIP but requires zero downloads and no GPU/torch dependency.
    """
    img = resize_image(img, IMAGE_SIZE)
    arr = np.asarray(img).astype(np.float32)

    hist = []
    for channel in range(3):
        h, _ = np.histogram(arr[:, :, channel], bins=32, range=(0, 255), density=True)
        hist.append(h)
    hist = np.concatenate(hist)

    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=0)).mean()
    gy = np.abs(np.diff(gray, axis=1)).mean()
    edge_density = np.array([gx, gy], dtype=np.float32)

    vec = np.concatenate([hist, edge_density]).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_image(path_or_image) -> np.ndarray:
    """
    Compute an embedding vector for a single image.
    Accepts a file path (str) or a PIL.Image.
    """
    img = load_image(path_or_image) if isinstance(path_or_image, str) else path_or_image

    if USE_CLIP:
        try:
            import torch
            model, processor = _load_clip()
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            vec = features[0].numpy().astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"[image_pipeline] CLIP unavailable ({e}); using fallback descriptor.")

    return _fallback_descriptor(img)


def embed_text_for_image_space(text: str) -> np.ndarray:
    """
    Embed a text description into the SAME space as embed_image, so image
    queries can be matched against KB text descriptions (CLIP supports this
    natively; fallback mode returns a zero vector sized to match, disabling
    cross-modal text->image matching gracefully).
    """
    if USE_CLIP:
        try:
            import torch
            model, processor = _load_clip()
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
            vec = features[0].numpy().astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"[image_pipeline] CLIP text embedding unavailable ({e}).")

    # fallback: cannot bridge text->image space meaningfully; return zeros
    dummy = _fallback_descriptor(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))
    return np.zeros_like(dummy)


if __name__ == "__main__":
    from data_loader import build_image_manifest
    manifest = build_image_manifest()
    if len(manifest):
        sample_path = manifest.iloc[0]["path"]
        vec = embed_image(sample_path)
        print("Embedding shape:", vec.shape)
    else:
        print("No images found. Run dataset_builder.py first.")
