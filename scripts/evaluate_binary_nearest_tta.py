
from pathlib import Path
import csv
import json

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
SIZE = (224, 224)
BATCH_SIZE = 16

CLASSES = ["Fresh", "Rotten"]

# Orientasi input yang diuji
TEST_ANGLES = [0, 45, 90, 135, 180, 270]

# Rotasi yang dirata-ratakan saat inference
TTA_ANGLES = [0, 90, 180, 270]

MODEL_PATH = Path(
    "models/binary_rotation_robust.keras"
)

VALID_DIR = Path("dataset/cleaned/valid")
EXTERNAL_DIR = Path("dataset/external_test/egg_001")

REPORT_DIR = Path(
    "reports/model_j/nearest_tta"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ==========================================
# IMAGE LOADING
# ==========================================

def load_validation_image(path):
    content = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        content,
        channels=3,
        expand_animations=False
    )

    image.set_shape((None, None, 3))

    image = tf.image.resize(
        image,
        SIZE,
        method="bilinear"
    )

    return image.numpy().astype(np.float32)


def load_external_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = np.asarray(
            img.convert("RGB")
        ).copy()

    return tf.image.resize(
        rgb,
        SIZE,
        method="bilinear"
    ).numpy().astype(np.float32)


def find_images(folder):
    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in (
            ".jpg", ".jpeg", ".png"
        )
    )


# ==========================================
# ROTATION WITH NEAREST FILL
# ==========================================

def rotate(image, angle):
    if angle == 0:
        return image.copy()

    if angle in (90, 180, 270):
        return np.rot90(
            image,
            k=angle // 90
        ).copy()

    height, width = image.shape[:2]

    center = (
        (width - 1) / 2,
        (height - 1) / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)


# ==========================================
# PREDICTION
# ==========================================

def predict_original(model, images):
    return model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=0
    )


def predict_tta(model, images):
    all_probabilities = []

    for angle in TTA_ANGLES:
        rotated = np.stack([
            rotate(img, angle)
            for img in images
        ]).astype(np.float32)

        probabilities = model.predict(
            rotated,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        all_probabilities.append(probabilities)

    return np.mean(
        np.stack(all_probabilities),
        axis=0
    )


# ==========================================
# EVALUATION
# ==========================================

def calculate_metrics(labels, probabilities):
    predicted = probabilities.argmax(axis=1)

    matrix = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1]
    )

    return {
        "accuracy": float(
            accuracy_score(labels, predicted)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),
        "fresh_recall": float(
            recall_score(
                labels,
                predicted,
                labels=[0],
                average="macro",
                zero_division=0
            )
        ),
        "rotten_recall": float(
            recall_score(
                labels,
                predicted,
                labels=[1],
                average="macro",
                zero_division=0
            )
        ),
        "confusion_matrix": matrix.tolist()
    }


def save_csv(path, records):
    if not records:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


# ==========================================
# MAIN
# ==========================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    print("\nLoading Model J...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    paths = []
    labels = []

    for class_id, class_name in enumerate(CLASSES):
        files = find_images(
            VALID_DIR / class_name
        )

        if not files:
            raise ValueError(
                f"Data kelas {class_name} tidak ditemukan"
            )

        paths.extend(files)
        labels.extend([class_id] * len(files))

    labels = np.asarray(
        labels,
        dtype=np.int32
    )

    images = np.stack([
        load_validation_image(path)
        for path in paths
    ]).astype(np.float32)

    print("Validation images:", len(images))

    summary = []
    prediction_rows = []

    # ======================================
    # VALIDATION ROTATION TEST
    # ======================================

    for angle in TEST_ANGLES:
        print(f"\nTesting input angle {angle}°...")

        input_images = np.stack([
            rotate(image, angle)
            for image in images
        ]).astype(np.float32)

        original_probs = predict_original(
            model,
            input_images
        )

        tta_probs = predict_tta(
            model,
            input_images
        )

        for method, probs in [
            ("original", original_probs),
            ("nearest_tta", tta_probs)
        ]:
            metrics = calculate_metrics(
                labels,
                probs
            )

            summary.append({
                "angle": angle,
                "method": method,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": (
                    metrics["confusion_matrix"]
                )
            })

            print(
                f"{method:12s} | "
                f"Accuracy {metrics['accuracy']:.2%} | "
                f"Macro F1 {metrics['macro_f1']:.2%} | "
                f"Rotten Recall {metrics['rotten_recall']:.2%}"
            )

            predicted = probs.argmax(axis=1)

            for i, path in enumerate(paths):
                prediction_rows.append({
                    "filename": str(
                        path.relative_to(VALID_DIR)
                    ),
                    "angle": angle,
                    "method": method,
                    "actual": CLASSES[labels[i]],
                    "prediction": CLASSES[predicted[i]],
                    "prob_fresh": float(probs[i, 0]),
                    "prob_rotten": float(probs[i, 1])
                })

    # ======================================
    # EXTERNAL PHOTOS
    # ======================================

    external_paths = find_images(EXTERNAL_DIR)
    external_records = []
    external_summary = {}

    if external_paths:
        external_images = np.stack([
            load_external_image(path)
            for path in external_paths
        ]).astype(np.float32)

        for method, probs in [
            (
                "original",
                predict_original(model, external_images)
            ),
            (
                "nearest_tta",
                predict_tta(model, external_images)
            )
        ]:
            predicted = probs.argmax(axis=1)

            agreement = (
                np.bincount(
                    predicted,
                    minlength=2
                ).max() / len(predicted)
            )

            external_summary[method] = {
                "majority_agreement": float(agreement),
                "distinct_classes": int(
                    len(set(predicted.tolist()))
                )
            }

            for i, path in enumerate(external_paths):
                external_records.append({
                    "filename": path.name,
                    "method": method,
                    "prediction": CLASSES[predicted[i]],
                    "prob_fresh": float(probs[i, 0]),
                    "prob_rotten": float(probs[i, 1])
                })

    # ======================================
    # SAVE RESULTS
    # ======================================

    save_csv(
        REPORT_DIR / "tta_comparison.csv",
        [
            {
                **row,
                "confusion_matrix": json.dumps(
                    row["confusion_matrix"]
                )
            }
            for row in summary
        ]
    )

    save_csv(
        REPORT_DIR / "tta_predictions.csv",
        prediction_rows
    )

    save_csv(
        REPORT_DIR / "external_tta_predictions.csv",
        external_records
    )

    averages = {}

    for method in ("original", "nearest_tta"):
        items = [
            row for row in summary
            if row["method"] == method
        ]

        averages[method] = {
            "mean_accuracy": float(np.mean([
                row["accuracy"] for row in items
            ])),
            "mean_macro_f1": float(np.mean([
                row["macro_f1"] for row in items
            ])),
            "mean_rotten_recall": float(np.mean([
                row["rotten_recall"] for row in items
            ]))
        }

    result = {
        "model": "binary_rotation_robust",
        "test_angles": TEST_ANGLES,
        "tta_angles": TTA_ANGLES,
        "validation_count": len(images),
        "rotations": summary,
        "averages": averages,
        "external_photo_count": len(external_paths),
        "external": external_summary,
        "notes": (
            "Exploratory evaluation on reused validation "
            "images. Physical egg freshness unverified."
        )
    }

    with open(
        REPORT_DIR / "tta_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    print("\n=== NEAREST TTA RESULTS ===")

    for method, values in averages.items():
        print(
            f"{method}: "
            f"Mean accuracy {values['mean_accuracy']:.2%}, "
            f"Mean F1 {values['mean_macro_f1']:.2%}, "
            f"Mean Rotten recall "
            f"{values['mean_rotten_recall']:.2%}"
        )

    print("\nExternal results:", external_summary)
    print("Saved:", REPORT_DIR)


if __name__ == "__main__":
    main()
