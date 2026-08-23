"""Loading utilities for the knowledge base, text dataset, and image manifest."""

import os
import glob
import pandas as pd

from config import KB_JSON_PATH, TEXT_DATASET_PATH, IMAGE_DIR, IMAGE_CATEGORIES
from utils import load_json


def load_knowledge_base() -> pd.DataFrame:
    """Load the structured airport knowledge base as a DataFrame."""
    records = load_json(KB_JSON_PATH)
    return pd.DataFrame(records)


def load_text_dataset() -> pd.DataFrame:
    """Load the labeled passenger query dataset (text + intent + entities + kb_id)."""
    records = load_json(TEXT_DATASET_PATH)
    return pd.DataFrame(records)


def build_image_manifest() -> pd.DataFrame:
    """Scan data/images/<category>/*.png and build a manifest DataFrame."""
    rows = []
    for category in IMAGE_CATEGORIES:
        folder = os.path.join(IMAGE_DIR, category)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
            rows.append({"path": path, "category": category, "filename": os.path.basename(path)})
    return pd.DataFrame(rows)


def train_val_split(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42):
    """Simple deterministic shuffle-and-split, stratification-free (small dataset)."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_df = shuffled.iloc[:n_val].reset_index(drop=True)
    train_df = shuffled.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


if __name__ == "__main__":
    kb = load_knowledge_base()
    text_df = load_text_dataset()
    img_df = build_image_manifest()
    print("KB records:", len(kb))
    print("Text queries:", len(text_df))
    print("Images found:", len(img_df))
