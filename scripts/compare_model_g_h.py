
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================
# CONFIGURATION
# ============================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = ["Fresh", "Half-Fresh", "Rotten"]
ANGLES = [0, 45, 90, 135, 180, 270]

MODEL_PATHS = {
    "Model_G": Path("models/rotation_robust.keras"),
    "Model_H": Path("models/domain_robust.keras"),
}

VALID_DIR = Path("dataset/cleaned/valid")
EXTERNAL_DIR = Path("dataset/external_test/egg_001")
REPORT_DIR = Path("reports/model_h/comparison")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# ============================================
# HELPERS
# ============================================

def save_csv(path, records):
    if not records:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB")).copy()


def resize_image(rgb):
    image = tf.image.resize(
        rgb,
        IMAGE_SIZE,
        method="bilinear"
    )
    return tf.cast(image, tf.float32)


def rotate_image(image, angle):
    # Agar reproduktif, pakai rotasi yang sama
    # untuk kedua model.
    # Angka positif = berlawanan arah jarum jam.
    if angle == 0:
        return image

    if angle == 90:
        return tf.image.rot90(image, k=1)

    if angle == 180:
        return tf.image.rot90(image, k=2)

    if angle == 270:
        return tf.image.rot90(image, k=3)

    # Untuk 45° dan 135°, gunakan PIL dengan
    # bilinear interpolation dan black fill.
    # Float 0..255 dipertahankan tanpa
    # MobileNetV2 preprocessing tambahan.
    array = tf.cast(
        tf.clip_by_value(image, 0, 255),
        tf.uint8
    ).numpy()

    pil_img = Image.fromarray(array, mode="RGB")

    pil_img = pil_img.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0)
    )

    return tf.convert_to_tensor(
        np.asarray(pil_img).copy(),
        dtype=tf.float32
    )


def load_validation():
    images = []
    labels = []
    names = []

    for label_id, class_name in enumerate(CLASS_NAMES):
        folder = VALID_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        paths = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".webp"
            )
        )

        for path in paths:
            rgb = read_image(path)
            images.append(resize_image(rgb).numpy())
            labels.append(label_id)
            names.append(str(path.relative_to(VALID_DIR)))

    if not images:
        raise ValueError("Validation dataset kosong")

    return (
        np.stack(images).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        names
    )


def load_external():
    if not EXTERNAL_DIR.is_dir():
        raise FileNotFoundError(EXTERNAL_DIR)

    paths = sorted(
        p for p in EXTERNAL_DIR.iterdir()
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".webp"
        )
    )

    if not paths:
        raise ValueError("Foto eksternal tidak ditemukan")

    images = np.stack([
        resize_image(read_image(p)).numpy()
        for p in paths
    ]).astype(np.float32)

    return images, [p.name for p in paths]


def predict_batch(model, images):
    return model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=0
    )


def compute_metrics(actual, predicted):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1, 2]
        ).tolist()
    }


# ============================================
# MAIN
# ============================================

def main():
    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} tidak ditemukan: {path}"
            )

    print("\nLoading validation images...")
    valid_images, valid_labels, valid_names = (
        load_validation()
    )

    print("Validation images:", len(valid_images))

    print("\nLoading external images...")
    external_images, external_names = load_external()

    print("External images:", len(external_images))

    # Siapkan hasil rotasi satu kali.
    # Kedua model memakai tensor yang identik.
    rotated_sets = {}

    for angle in ANGLES:
        print(f"Preparing rotation {angle}°...")

        rotated_sets[angle] = np.stack([
            rotate_image(
                tf.convert_to_tensor(img),
                angle
            ).numpy()
            for img in valid_images
        ]).astype(np.float32)

    summary_rows = []
    prediction_rows = []
    external_rows = []
    full_metrics = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n=== EVALUATING {model_name} ===")

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        model_results = []

        for angle in ANGLES:
            images = rotated_sets[angle]

            probabilities = predict_batch(
                model,
                images
            )

            predicted = probabilities.argmax(axis=1)

            metrics = compute_metrics(
                valid_labels,
                predicted
            )

            summary_rows.append({
                "model": model_name,
                "angle": angle,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            })

            model_results.append({
                "angle": angle,
                **metrics
            })

            print(
                f"{angle:3d}° | "
                f"Accuracy: {metrics['accuracy']:.2%} | "
                f"Macro F1: {metrics['macro_f1']:.2%}"
            )

            if angle == 0:
                report = classification_report(
                    valid_labels,
                    predicted,
                    labels=[0, 1, 2],
                    target_names=CLASS_NAMES,
                    zero_division=0
                )

                report_path = (
                    REPORT_DIR /
                    f"{model_name.lower()}_classification_report.txt"
                )

                report_path.write_text(
                    report,
                    encoding="utf-8"
                )

            for i, filename in enumerate(valid_names):
                prediction_rows.append({
                    "model": model_name,
                    "angle": angle,
                    "filename": filename,
                    "actual": CLASS_NAMES[valid_labels[i]],
                    "prediction": CLASS_NAMES[predicted[i]],
                    "confidence": float(
                        probabilities[i, predicted[i]] * 100
                    )
                })

        # Evaluasi foto fisik asli, tanpa rotasi TTA.
        external_probs = predict_batch(
            model,
            external_images
        )

        external_pred = external_probs.argmax(axis=1)

        for i, filename in enumerate(external_names):
            record = {
                "model": model_name,
                "filename": filename,
                "prediction": CLASS_NAMES[external_pred[i]],
                "confidence": float(
                    external_probs[i, external_pred[i]] * 100
                )
            }

            for class_id, label in enumerate(CLASS_NAMES):
                record[f"prob_{label}"] = float(
                    external_probs[i, class_id] * 100
                )

            external_rows.append(record)

            print(
                f"{filename}: "
                f"{record['prediction']} "
                f"({record['confidence']:.2f}%)"
            )

        majority_agreement = float(
            np.bincount(
                external_pred,
                minlength=len(CLASS_NAMES)
            ).max() / len(external_pred)
        )

        full_metrics[model_name] = {
            "rotations": model_results,
            "mean_rotation_accuracy": float(
                np.mean([
                    item["accuracy"]
                    for item in model_results
                ])
            ),
            "mean_rotation_macro_f1": float(
                np.mean([
                    item["macro_f1"]
                    for item in model_results
                ])
            ),
            "external_photo_count": len(external_names),
            "external_distinct_classes": int(
                len(set(external_pred.tolist()))
            ),
            "external_majority_agreement": (
                majority_agreement
            )
        }

        print(
            "Mean rotation accuracy:",
            f"{full_metrics[model_name]['mean_rotation_accuracy']:.2%}"
        )
        print(
            "External majority agreement:",
            f"{majority_agreement:.2%}"
        )

        del model
        tf.keras.backend.clear_session()

    save_csv(
        REPORT_DIR / "rotation_comparison.csv",
        summary_rows
    )

    save_csv(
        REPORT_DIR / "rotation_predictions.csv",
        prediction_rows
    )

    save_csv(
        REPORT_DIR / "external_predictions.csv",
        external_rows
    )

    save_json(
        REPORT_DIR / "comparison_metrics.json",
        full_metrics
    )

    print("\n=== COMPARISON FINISHED ===")
    print("Saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
