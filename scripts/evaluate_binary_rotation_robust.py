
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path(
    "models/binary_rotation_robust.keras"
)

VALID_DIR = Path("dataset/cleaned/valid")
EXTERNAL_DIR = Path("dataset/external_test/egg_001")
REPORT_DIR = Path("reports/model_j/evaluation")

CLASS_NAMES = ["Fresh", "Rotten"]
ANGLES = [0, 45, 90, 135, 180, 270]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# IMAGE LOADING
# ==========================================

def load_tf_image(path):
    content = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        content,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    return tf.cast(image, tf.float32).numpy()


def load_external_image(path):
    # File HP memakai EXIF orientation.
    # Catat bahwa decoder ini berbeda dari
    # evaluasi validation set TensorFlow.
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = np.asarray(img.convert("RGB")).copy()

    return tf.image.resize(
        rgb,
        IMAGE_SIZE,
        method="bilinear"
    ).numpy().astype(np.float32)


def get_paths(folder):
    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png"
        )
    )


# ==========================================
# ROTATION
# ==========================================

def rotate_image(image, angle):
    if angle in [0, 90, 180, 270]:
        return np.rot90(
            image,
            k=angle // 90
        ).copy()

    # 45° dan 135° memakai interpolasi PIL.
    pil = Image.fromarray(
        np.clip(image, 0, 255).astype(np.uint8)
    )

    rotated = pil.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0)
    )

    return np.asarray(rotated).astype(np.float32)


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(actual, probabilities):
    predicted = probabilities.argmax(axis=1)

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_precision": float(
            precision_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            )
        ),
        "fresh_recall": float(
            recall_score(
                actual,
                predicted,
                labels=[0],
                average="macro",
                zero_division=0
            )
        ),
        "rotten_recall": float(
            recall_score(
                actual,
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
        path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    paths = []
    actual = []

    for label_id, class_name in enumerate(CLASS_NAMES):
        class_paths = get_paths(
            VALID_DIR / class_name
        )

        if not class_paths:
            raise ValueError(
                f"Data {class_name} tidak ditemukan."
            )

        paths.extend(class_paths)
        actual.extend(
            [label_id] * len(class_paths)
        )

    actual = np.asarray(actual, dtype=np.int32)

    images = np.stack(
        [load_tf_image(path) for path in paths]
    ).astype(np.float32)

    # ======================================
    # SIX-ANGLE ROTATION EVALUATION
    # ======================================

    rotation_results = []
    prediction_records = []

    print("\n=== ROTATION EVALUATION ===")

    for angle in ANGLES:
        rotated_images = np.stack([
            rotate_image(image, angle)
            for image in images
        ]).astype(np.float32)

        probabilities = model.predict(
            rotated_images,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        metrics = calculate_metrics(
            actual,
            probabilities
        )

        rotation_results.append({
            "angle": angle,
            **metrics
        })

        predictions = probabilities.argmax(axis=1)

        for index, path in enumerate(paths):
            prediction_records.append({
                "angle": angle,
                "filename": str(
                    path.relative_to(VALID_DIR)
                ),
                "actual": CLASS_NAMES[actual[index]],
                "prediction": CLASS_NAMES[predictions[index]],
                "prob_fresh": float(
                    probabilities[index, 0]
                ),
                "prob_rotten": float(
                    probabilities[index, 1]
                )
            })

        print(
            f"{angle:3d}° | "
            f"Accuracy: {metrics['accuracy']:.2%} | "
            f"Macro F1: {metrics['macro_f1']:.2%}"
        )

    save_csv(
        REPORT_DIR / "rotation_predictions.csv",
        prediction_records
    )

    # ======================================
    # HALF-FRESH PROBE
    # ======================================

    half_paths = get_paths(
        VALID_DIR / "Half-Fresh"
    )

    half_counts = {"Fresh": 0, "Rotten": 0}
    half_records = []

    if half_paths:
        half_images = np.stack([
            load_tf_image(path)
            for path in half_paths
        ]).astype(np.float32)

        half_probs = model.predict(
            half_images,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        for index, path in enumerate(half_paths):
            predicted = int(
                np.argmax(half_probs[index])
            )
            name = CLASS_NAMES[predicted]

            half_counts[name] += 1

            half_records.append({
                "filename": path.name,
                "prediction": name,
                "prob_fresh": float(
                    half_probs[index, 0]
                ),
                "prob_rotten": float(
                    half_probs[index, 1]
                )
            })

    save_csv(
        REPORT_DIR / "half_fresh_predictions.csv",
        half_records
    )

    # ======================================
    # EXTERNAL PHYSICAL EGG PHOTOS
    # ======================================

    external_paths = get_paths(EXTERNAL_DIR)
    external_records = []
    external_labels = []

    if external_paths:
        external_images = np.stack([
            load_external_image(path)
            for path in external_paths
        ]).astype(np.float32)

        external_probs = model.predict(
            external_images,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        for index, path in enumerate(external_paths):
            predicted = int(
                np.argmax(external_probs[index])
            )

            label = CLASS_NAMES[predicted]
            external_labels.append(label)

            external_records.append({
                "filename": path.name,
                "prediction": label,
                "prob_fresh": float(
                    external_probs[index, 0]
                ),
                "prob_rotten": float(
                    external_probs[index, 1]
                )
            })

    save_csv(
        REPORT_DIR / "external_predictions.csv",
        external_records
    )

    agreement = None

    if external_labels:
        agreement = max(
            external_labels.count(name)
            for name in CLASS_NAMES
        ) / len(external_labels)

    # ======================================
    # SAVE RESULTS
    # ======================================

    mean_accuracy = float(np.mean([
        result["accuracy"]
        for result in rotation_results
    ]))

    mean_f1 = float(np.mean([
        result["macro_f1"]
        for result in rotation_results
    ]))

    result = {
        "model": "binary_rotation_robust",
        "validation_count": len(paths),
        "normal_validation": rotation_results[0],
        "rotations": rotation_results,
        "mean_rotation_accuracy": mean_accuracy,
        "mean_rotation_macro_f1": mean_f1,
        "half_fresh_count": len(half_paths),
        "half_fresh_predictions": half_counts,
        "external_photo_count": len(external_paths),
        "external_majority_agreement": agreement
    }

    with open(
        REPORT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    # Klasifikasi normal tanpa rotasi.
    original_predictions = [
        CLASS_NAMES.index(row["prediction"])
        for row in prediction_records
        if row["angle"] == 0
    ]

    report = classification_report(
        actual,
        original_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    (
        REPORT_DIR / "classification_report.txt"
    ).write_text(report, encoding="utf-8")

    print("\n=== MODEL J SUMMARY ===")
    print(f"Mean rotation accuracy: {mean_accuracy:.2%}")
    print(f"Mean rotation macro F1: {mean_f1:.2%}")
    print("Half-Fresh:", half_counts)

    if agreement is not None:
        print(
            f"External majority agreement: "
            f"{agreement:.2%}"
        )

    print("Reports saved:", REPORT_DIR)


if __name__ == "__main__":
    main()
