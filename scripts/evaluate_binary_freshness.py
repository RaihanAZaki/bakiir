
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = Path("models/binary_freshness.keras")

VALID_DIR = Path("dataset/cleaned/valid")
EXTERNAL_DIR = Path("dataset/external_test/egg_001")

REPORT_DIR = Path("reports/model_i/evaluation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Fresh", "Rotten"]
IMAGE_SIZE = (224, 224)

# Tidak memakai test set lama untuk
# pemilihan model atau tuning tambahan.


# ============================================
# IMAGE PROCESSING
# ============================================

def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = np.asarray(
            img.convert("RGB")
        ).copy()

    image = tf.image.resize(
        rgb,
        IMAGE_SIZE,
        method="bilinear"
    )

    # Model I memiliki Rescaling internal.
    # Jangan preprocessing -1..1 lagi.
    return tf.cast(
        image,
        tf.float32
    ).numpy()


def get_images(folder):
    if not folder.is_dir():
        return [], []

    paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png"
        )
    )

    if not paths:
        return [], []

    images = np.stack(
        [load_image(path) for path in paths]
    ).astype(np.float32)

    return paths, images


def save_csv(path, records):
    if not records:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    records = []

    # ========================================
    # VALIDATION: FRESH AND ROTTEN
    # ========================================

    actual = []
    predicted = []

    for label_id, label_name in enumerate(CLASS_NAMES):
        paths, images = get_images(
            VALID_DIR / label_name
        )

        if not paths:
            raise ValueError(
                f"Tidak ditemukan gambar {label_name}"
            )

        probs = model.predict(
            images,
            batch_size=16,
            verbose=0
        )

        for i, path in enumerate(paths):
            prediction_id = int(
                np.argmax(probs[i])
            )

            actual.append(label_id)
            predicted.append(prediction_id)

            records.append({
                "dataset": "validation",
                "filename": path.name,
                "actual": label_name,
                "prediction": CLASS_NAMES[prediction_id],
                "prob_fresh": float(probs[i, 0]),
                "prob_rotten": float(probs[i, 1])
            })

    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=[0, 1],
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    # ========================================
    # HALF-FRESH: OUTSIDE TWO-CLASS TASK
    # ========================================

    half_paths, half_images = get_images(
        VALID_DIR / "Half-Fresh"
    )

    half_counts = {"Fresh": 0, "Rotten": 0}

    if half_paths:
        half_probs = model.predict(
            half_images,
            batch_size=16,
            verbose=0
        )

        for i, path in enumerate(half_paths):
            idx = int(np.argmax(half_probs[i]))
            label = CLASS_NAMES[idx]

            half_counts[label] += 1

            records.append({
                "dataset": "half_fresh_probe",
                "filename": path.name,
                "actual": "Half-Fresh",
                "prediction": label,
                "prob_fresh": float(half_probs[i, 0]),
                "prob_rotten": float(half_probs[i, 1])
            })

    # ========================================
    # EXTERNAL PHYSICAL EGG PHOTOS
    # ========================================

    external_paths, external_images = get_images(
        EXTERNAL_DIR
    )

    external_predictions = []

    if external_paths:
        external_probs = model.predict(
            external_images,
            batch_size=16,
            verbose=0
        )

        for i, path in enumerate(external_paths):
            idx = int(np.argmax(external_probs[i]))
            label = CLASS_NAMES[idx]

            external_predictions.append(label)

            records.append({
                "dataset": "external",
                "filename": path.name,
                "actual": "Unknown",
                "prediction": label,
                "prob_fresh": float(external_probs[i, 0]),
                "prob_rotten": float(external_probs[i, 1])
            })

    consistency = None

    if external_predictions:
        consistency = (
            max(
                external_predictions.count(label)
                for label in CLASS_NAMES
            ) / len(external_predictions)
        )

    # ========================================
    # SAVE REPORTS
    # ========================================

    results = {
        "validation_count": len(actual),
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "half_fresh_count": len(half_paths),
        "half_fresh_predicted_as": half_counts,
        "external_photo_count": len(external_paths),
        "external_majority_agreement": consistency,
        "limitations": [
            "Half-Fresh has no target class in Model I.",
            "External photos have no verified freshness label.",
            "Validation data have been used during development.",
            "Egg identity independence is not established."
        ]
    }

    with open(
        REPORT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    save_csv(
        REPORT_DIR / "all_predictions.csv",
        records
    )

    (REPORT_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    print("\n=== MODEL I EVALUATION ===")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Validation macro F1: {macro_f1:.2%}")
    print("\nConfusion matrix:")
    print(matrix)

    print("\nHalf-Fresh predictions:")
    print(half_counts)

    print("\nExternal majority agreement:")
    if consistency is None:
        print("No external photos")
    else:
        print(f"{consistency:.2%}")

    print("\nSaved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
