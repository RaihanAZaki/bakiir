
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = Path("models/binary_freshness.keras")
VALID_DIR = Path("dataset/cleaned/valid")
REPORT_DIR = Path("reports/model_i/preprocessing_audit")

CLASS_NAMES = ["Fresh", "Rotten"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# LOAD FILE PATHS
# ============================================

def get_validation_files():
    paths = []
    labels = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        folder = VALID_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in [
                ".jpg", ".jpeg", ".png"
            ]
        )

        for path in files:
            paths.append(path)
            labels.append(class_id)

    return paths, np.asarray(labels, dtype=np.int32)


# ============================================
# PREPROCESSING A: TENSORFLOW
# ============================================

def preprocess_tensorflow(path):
    content = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        content,
        channels=3,
        expand_animations=False
    )

    image.set_shape((None, None, 3))

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    return tf.cast(
        image,
        tf.float32
    ).numpy()


# ============================================
# PREPROCESSING B: PIL
# ============================================

def preprocess_pil(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        image = np.asarray(
            img.convert("RGB")
        ).copy()

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    return tf.cast(
        image,
        tf.float32
    ).numpy()


# ============================================
# METRICS
# ============================================

def evaluate(actual, probabilities):
    predicted = probabilities.argmax(axis=1)

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).tolist()
    }


# ============================================
# MAIN
# ============================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    paths, labels = get_validation_files()

    if not paths:
        raise ValueError("Validation dataset kosong.")

    print("\n=== MODEL I PREPROCESSING AUDIT ===")
    print("Model:", MODEL_PATH)
    print("Total images:", len(paths))

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Processing images with TensorFlow...")

    tf_images = np.stack([
        preprocess_tensorflow(path)
        for path in paths
    ]).astype(np.float32)

    print("Processing images with PIL...")

    pil_images = np.stack([
        preprocess_pil(path)
        for path in paths
    ]).astype(np.float32)

    print("Running TensorFlow predictions...")

    tf_probs = model.predict(
        tf_images,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print("Running PIL predictions...")

    pil_probs = model.predict(
        pil_images,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    tf_predictions = tf_probs.argmax(axis=1)
    pil_predictions = pil_probs.argmax(axis=1)

    tf_metrics = evaluate(labels, tf_probs)
    pil_metrics = evaluate(labels, pil_probs)

    pixel_difference = np.abs(
        tf_images - pil_images
    )

    rows = []

    for i, path in enumerate(paths):
        pixel_mae = float(
            np.mean(pixel_difference[i])
        )

        pixel_max_diff = float(
            np.max(pixel_difference[i])
        )

        prediction_changed = bool(
            tf_predictions[i] != pil_predictions[i]
        )

        with Image.open(path) as img:
            exif_orientation = img.getexif().get(274, 1)

        rows.append({
            "filename": str(path.relative_to(VALID_DIR)),
            "actual": CLASS_NAMES[labels[i]],
            "tensorflow_prediction": (
                CLASS_NAMES[tf_predictions[i]]
            ),
            "tensorflow_fresh_probability": float(
                tf_probs[i, 0]
            ),
            "tensorflow_rotten_probability": float(
                tf_probs[i, 1]
            ),
            "pil_prediction": (
                CLASS_NAMES[pil_predictions[i]]
            ),
            "pil_fresh_probability": float(
                pil_probs[i, 0]
            ),
            "pil_rotten_probability": float(
                pil_probs[i, 1]
            ),
            "pixel_mae": pixel_mae,
            "pixel_max_difference": pixel_max_diff,
            "exif_orientation": exif_orientation,
            "prediction_changed": prediction_changed
        })

    changed_rows = [
        row for row in rows
        if row["prediction_changed"]
    ]

    # ========================================
    # SAVE CSV
    # ========================================

    csv_path = REPORT_DIR / "preprocessing_comparison.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    # ========================================
    # SAVE JSON
    # ========================================

    summary = {
        "model": str(MODEL_PATH),
        "total_images": len(paths),
        "tensorflow": tf_metrics,
        "pil": pil_metrics,
        "prediction_changed_count": len(changed_rows),
        "mean_pixel_mae": float(
            np.mean(pixel_difference)
        ),
        "max_pixel_difference": float(
            np.max(pixel_difference)
        ),
        "exactly_identical_image_count": int(
            np.count_nonzero(
                np.all(
                    tf_images == pil_images,
                    axis=(1, 2, 3)
                )
            )
        )
    }

    json_path = REPORT_DIR / "audit_summary.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n=== AUDIT RESULTS ===")

    print(
        "TensorFlow accuracy:",
        f"{tf_metrics['accuracy']:.2%}"
    )

    print(
        "PIL accuracy:",
        f"{pil_metrics['accuracy']:.2%}"
    )

    print(
        "Changed predictions:",
        len(changed_rows)
    )

    print(
        "Mean pixel MAE:",
        f"{summary['mean_pixel_mae']:.6f}"
    )

    print("\nChanged images:")

    for row in changed_rows:
        print(
            row["filename"],
            "| TF:", row["tensorflow_prediction"],
            "| PIL:", row["pil_prediction"],
            "| Pixel MAE:", f"{row['pixel_mae']:.4f}"
        )

    print("\nFiles saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
