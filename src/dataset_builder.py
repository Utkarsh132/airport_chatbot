"""
Synthetic image dataset generator.

Because we cannot rely on proprietary or copyrighted airport photography,
this script generates simple synthetic "sign-style" images for each
airport category (gate, baggage_claim, check_in, security, restroom,
lounge, transport, information_desk, restaurant).

Each image is a colored icon-and-text card that mimics a real airport
sign closely enough for a proof-of-concept vision pipeline (CLIP
embeddings respond well to iconography + text layout).

LIMITATIONS (documented, as required by the assignment):
- These are synthetic placeholders, not real-world photographs.
- No lighting variation, camera noise, or perspective distortion is modeled.
- Real deployment would require photographs of actual airport signage
  under varied lighting/angles for the vision model to generalize.
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

from config import IMAGE_DIR, IMAGE_CATEGORIES, RANDOM_SEED

random.seed(RANDOM_SEED)

CATEGORY_COLORS = {
    "gate": (0, 105, 111),
    "baggage_claim": (150, 66, 25),
    "check_in": (1, 105, 111),
    "security": (161, 44, 123),
    "restroom": (0, 100, 148),
    "lounge": (122, 57, 187),
    "transport": (67, 122, 34),
    "information_desk": (209, 153, 0),
    "restaurant": (218, 113, 1),
}

CATEGORY_LABELS = {
    "gate": ["GATE A5", "GATE B12", "GATE C3", "GATE D7"],
    "baggage_claim": ["BAGGAGE CLAIM", "CAROUSEL 3", "LUGGAGE"],
    "check_in": ["CHECK-IN", "BAG DROP", "TICKETING"],
    "security": ["SECURITY", "PASSPORT CONTROL", "SCREENING"],
    "restroom": ["RESTROOM", "WC", "TOILETS"],
    "lounge": ["LOUNGE", "PRAYER ROOM", "REST AREA"],
    "transport": ["TAXI", "TRAIN", "SHUTTLE BUS"],
    "information_desk": ["INFO DESK", "ASSISTANCE", "HELP POINT"],
    "restaurant": ["RESTAURANT", "FOOD COURT", "CAFE"],
}


def _get_font(size=28):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def generate_synthetic_image(category: str, index: int, size=(224, 224)) -> Image.Image:
    color = CATEGORY_COLORS.get(category, (100, 100, 100))
    label = random.choice(CATEGORY_LABELS.get(category, [category.upper()]))

    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)

    # simple icon: rounded rectangle + circle to vary layout deterministically per category
    draw.rounded_rectangle([20, 20, size[0] - 20, size[1] - 60], radius=16,
                            outline=(255, 255, 255), width=4)
    draw.ellipse([size[0] // 2 - 25, 40, size[0] // 2 + 25, 90], fill=(255, 255, 255))

    font = _get_font(20)
    text_w = draw.textlength(label, font=font)
    draw.text(((size[0] - text_w) / 2, size[1] - 45), label, fill=(255, 255, 255), font=font)

    return img


def build_dataset(images_per_category: int = 8):
    """Generate a small synthetic dataset and save under data/images/<category>/."""
    manifest = []
    for category in IMAGE_CATEGORIES:
        out_dir = os.path.join(IMAGE_DIR, category)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(images_per_category):
            img = generate_synthetic_image(category, i)
            filename = f"{category}_{i:02d}.png"
            path = os.path.join(out_dir, filename)
            img.save(path)
            manifest.append({"path": path, "category": category, "filename": filename})
    return manifest


if __name__ == "__main__":
    manifest = build_dataset(images_per_category=8)
    print(f"Generated {len(manifest)} synthetic images across {len(IMAGE_CATEGORIES)} categories.")
