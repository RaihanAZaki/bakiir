
from pathlib import Path
import csv
import json

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# =====================================
# CONFIGURATION
# =====================================

MODEL_PATH = Path("models/rotation_robust.keras")
VALID_DIR = Path("dataset/cleaned/valid")
REPORT_DIR = Path("reports/model_h/validation")

CLASS_NAMES = ["Fresh", "Half-Fresh", "Rotten"]

IMAGE_SIZE = (224, 224)
MASK_SIZE = 256
GRABCUT_ITERATIONS = 3

REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(42)

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# =====================================
# IMAGE LOADING
# =====================================

def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(
            img.convert("RGB")
        ).copy()


# =====================================
# SEGMENTATION
# =====================================

def segment_egg(rgb):
    height, width = rgb.shape[:2]

    # Segmentasi pada ukuran lebih kecil
    # agar pengujian seluruh dataset lebih cepat.
    small = cv2.resize(
        rgb,
        (MASK_SIZE, MASK_SIZE),
        interpolation=cv2.INTER_AREA
    )

    h, w = small.shape[:2]

    mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    border = max(2, int(min(h, w) * 0.03))

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    center = (w // 2, h // 2)

    cv2.ellipse(
        mask,
        center,
        (int(w * 0.38), int(h * 0.38)),
        0, 0, 360,
        cv2.GC_PR_FGD,
        -1
    )

    cv2.ellipse(
        mask,
        center,
        (int(w * 0.10), int(h * 0.10)),
        0, 0, 360,
        cv2.GC_FGD,
        -1
    )

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    bgr = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2BGR
    )

    cv2.grabCut(
        bgr,
        mask,
        None,
        bg_model,
        fg_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK
    )

    binary = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        raise ValueError(
            "Tidak ditemukan foreground"
        )

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    binary = np.where(
        labels == largest,
        255,
        0
    ).astype(np.uint8)

    area_ratio = (
        np.count_nonzero(binary) / binary.size
    )

    if not 0.03 <= area_ratio <= 0.90:
        raise ValueError(
            f"Area mask tidak wajar: {area_ratio:.2%}"
        )

    # Kembalikan mask ke ukuran asli.
    binary = cv2.resize(
        binary,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    segmented = np.zeros_like(rgb)
    segmented[binary > 0] = rgb[binary > 0]

    return segmented, binary


# =====================================
# MODEL PREDICTION
# =====================================

def prepare_image(rgb):
    image = tf.image.resize(
        rgb,
        IMAGE_SIZE,
        method="bilinear"
    )

    # MobileNetV2 preprocessing sudah
    # berada di dalam model.
    return tf.cast(
        image,
        tf.float32
    ).numpy()


def predict(rgb):
    image = prepare_image(rgb)

    probabilities = model.predict(
        image[None, ...],
        verbose=0
    )[0]

    index = int(np.argmax(probabilities))

    return index, probabilities


# =====================================
# METRICS
# =====================================

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1, 2]
            ).tolist()
        )
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


# =====================================
# PREVIEW GENERATOR
# =====================================

def make_preview(samples):
    if not samples:
        return

    cell_w = 240
    cell_h = 260

    sheet = Image.new(
        "RGB",
        (cell_w * 3, cell_h * len(samples) + 40),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for col, title in enumerate(
        ["Original", "Mask", "Segmented"]
    ):
        draw.text(
            (col * cell_w + 10, 12),
            title,
            fill="black"
        )

    for row, sample in enumerate(samples):
        filename, original, mask, segmented = sample

        pictures = [
            Image.fromarray(original),
            Image.fromarray(mask).convert("RGB"),
            Image.fromarray(segmented)
        ]

        y = 40 + row * cell_h

        for col, picture in enumerate(pictures):
            thumbnail = ImageOps.contain(
                picture,
                (220, 215)
            )

            sheet.paste(
                thumbnail,
                (col * cell_w + 10, y)
            )

        draw.text(
            (10, y + 220),
            filename[:36],
            fill="black"
        )

    sheet.save(
        REPORT_DIR / "validation_preview.jpg",
        quality=95
    )


# =====================================
# MAIN EVALUATION
# =====================================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not VALID_DIR.exists():
        raise FileNotFoundError(VALID_DIR)

    image_paths = []

    for label_id, class_name in enumerate(CLASS_NAMES):
        class_dir = VALID_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)

        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in (
                ".jpg", ".jpeg", ".png",
                ".webp"
            ):
                image_paths.append(
                    (path, label_id)
                )

    print("\n=== MODEL H.3 VALIDATION ===")
    print("Total images:", len(image_paths))

    if not image_paths:
        raise ValueError(
            "Validation dataset kosong."
        )

    records = []
    preview_samples = []

    true_labels = []
    original_predictions = []
    segmented_predictions = []

    failed_count = 0

    for number, (path, label_id) in enumerate(
        image_paths,
        start=1
    ):
        rgb = load_image(path)

        original_id, original_probs = predict(rgb)

        segmentation_ok = True

        try:
            segmented, binary = segment_egg(rgb)

        except (cv2.error, ValueError) as exc:
            # Tetap catat original, tetapi jangan
            # menganggap fallback sebagai
            # hasil segmentasi yang valid.
            segmentation_ok = False
            failed_count += 1

            segmented = None
            binary = None

            print(
                f"Segmentation failed: {path.name}: {exc}"
            )

        record = {
            "filename": str(
                path.relative_to(VALID_DIR)
            ),
            "actual": CLASS_NAMES[label_id],
            "original_prediction": (
                CLASS_NAMES[original_id]
            ),
            "original_confidence": float(
                np.max(original_probs) * 100
            ),
            "segmentation_success": segmentation_ok,
            "segmented_prediction": "",
            "segmented_confidence": "",
            "prediction_changed": ""
        }

        if segmentation_ok:
            segmented_id, segmented_probs = predict(
                segmented
            )

            record["segmented_prediction"] = (
                CLASS_NAMES[segmented_id]
            )

            record["segmented_confidence"] = float(
                np.max(segmented_probs) * 100
            )

            record["prediction_changed"] = (
                original_id != segmented_id
            )

            true_labels.append(label_id)
            original_predictions.append(original_id)
            segmented_predictions.append(segmented_id)

            # Ambil preview dari setiap kelas.
            if len(preview_samples) < 12:
                preview_samples.append(
                    (
                        str(path.relative_to(VALID_DIR)),
                        rgb,
                        binary,
                        segmented
                    )
                )

        records.append(record)

        if number % 20 == 0 or number == len(image_paths):
            print(
                f"Processed {number}/{len(image_paths)}"
            )

    save_csv(
        REPORT_DIR / "validation_predictions.csv",
        records
    )

    # Evaluasi original pada seluruh validasi.
    all_actual = [
        CLASS_NAMES.index(record["actual"])
        for record in records
    ]

    all_original = [
        CLASS_NAMES.index(
            record["original_prediction"]
        )
        for record in records
    ]

    original_full_metrics = calculate_metrics(
        all_actual,
        all_original
    )

    if not true_labels:
        raise RuntimeError(
            "Seluruh proses segmentasi gagal."
        )

    # Perbandingan yang adil hanya pada
    # subset di mana segmentasi berhasil.
    original_subset_metrics = calculate_metrics(
        true_labels,
        original_predictions
    )

    segmented_metrics = calculate_metrics(
        true_labels,
        segmented_predictions
    )

    changed = sum(
        a != b
        for a, b in zip(
            original_predictions,
            segmented_predictions
        )
    )

    result = {
        "total_images": len(image_paths),
        "segmentation_success": len(true_labels),
        "segmentation_failed": failed_count,
        "prediction_changed": changed,
        "original_full": original_full_metrics,
        "original_on_successful_subset": (
            original_subset_metrics
        ),
        "segmented_on_successful_subset": (
            segmented_metrics
        )
    }

    with open(
        REPORT_DIR / "validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    for method, predictions in [
        ("original", original_predictions),
        ("segmented", segmented_predictions)
    ]:
        report = classification_report(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )

        with open(
            REPORT_DIR / f"{method}_report.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(report)

    make_preview(preview_samples)

    print("\n=== RESULTS ===")
    print(
        "Original full accuracy:",
        f"{original_full_metrics['accuracy']:.2%}"
    )
    print(
        "Original subset accuracy:",
        f"{original_subset_metrics['accuracy']:.2%}"
    )
    print(
        "Segmented subset accuracy:",
        f"{segmented_metrics['accuracy']:.2%}"
    )
    print(
        "Original subset macro F1:",
        f"{original_subset_metrics['macro_f1']:.2%}"
    )
    print(
        "Segmented subset macro F1:",
        f"{segmented_metrics['macro_f1']:.2%}"
    )
    print(
        "Changed predictions:",
        changed
    )
    print(
        "Segmentation failures:",
        failed_count
    )
    print(
        "\nReports saved to:",
        REPORT_DIR
    )


if __name__ == "__main__":
    main()
