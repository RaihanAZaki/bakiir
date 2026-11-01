
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageOps

MODEL_PATH = Path("models/rotation_robust.keras")
IMAGE_DIR = Path("dataset/external_test/egg_001")
REPORT_DIR = Path("reports/model_h")

CLASS_NAMES = ["Fresh", "Half-Fresh", "Rotten"]
ANGLES = [0, 90, 180, 270]
IMG_SIZE = (224, 224)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        image = np.asarray(img.convert("RGB"))

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method="bilinear"
    )

    return tf.cast(image, tf.float32)


def predict_original(image):
    batch = tf.expand_dims(image, axis=0)

    return model.predict(
        batch,
        verbose=0
    )[0]


def predict_tta(image):
    variants = []

    for k in range(4):
        rotated = tf.image.rot90(image, k=k)
        variants.append(rotated.numpy())

    batch = np.stack(
        variants,
        axis=0
    ).astype(np.float32)

    predictions = model.predict(
        batch,
        verbose=0
    )

    # Rata-rata probabilitas empat orientasi
    return np.mean(predictions, axis=0)


records = []

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

if not image_paths:
    raise ValueError("Tidak ada gambar pengujian.")

for path in image_paths:
    image = load_image(path)

    original_probs = predict_original(image)
    tta_probs = predict_tta(image)

    original_id = int(np.argmax(original_probs))
    tta_id = int(np.argmax(tta_probs))

    record = {
        "filename": path.name,
        "original_prediction": CLASS_NAMES[original_id],
        "original_confidence": float(
            original_probs[original_id] * 100
        ),
        "tta_prediction": CLASS_NAMES[tta_id],
        "tta_confidence": float(
            tta_probs[tta_id] * 100
        )
    }

    for i, label in enumerate(CLASS_NAMES):
        record[f"tta_{label}"] = float(
            tta_probs[i] * 100
        )

    records.append(record)

    print(f"\nFoto: {path.name}")
    print(
        f"Original: {record['original_prediction']} "
        f"({record['original_confidence']:.2f}%)"
    )
    print(
        f"TTA: {record['tta_prediction']} "
        f"({record['tta_confidence']:.2f}%)"
    )

df = pd.DataFrame(records)

df.to_csv(
    REPORT_DIR / "model_h_predictions.csv",
    index=False
)

original_consistency = (
    df["original_prediction"].value_counts().max()
    / len(df)
)

tta_consistency = (
    df["tta_prediction"].value_counts().max()
    / len(df)
)

summary = pd.DataFrame([
    {
        "method": "original",
        "majority_agreement": original_consistency
    },
    {
        "method": "tta",
        "majority_agreement": tta_consistency
    }
])

summary.to_csv(
    REPORT_DIR / "model_h_consistency.csv",
    index=False
)

print("\n=== KONSISTENSI ANTAR FOTO ===")
print(summary.to_string(index=False))

print("\nEksperimen Model H selesai.")
