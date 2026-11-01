
from pathlib import Path
import csv
import json

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = ["Fresh", "Rotten"]
ANGLES = [0, 45, 90, 135, 180, 270]

FILL_METHODS = {
    "black_fill": cv2.BORDER_CONSTANT,
    "white_fill": cv2.BORDER_CONSTANT,
    "nearest_fill": cv2.BORDER_REPLICATE,
}

MODEL_PATH = Path(
    "models/binary_rotation_robust.keras"
)

VALID_DIR = Path("dataset/cleaned/valid")

REPORT_DIR = Path(
    "reports/model_j/rotation_diagnostic"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

def get_files():
    paths = []
    labels = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        folder = VALID_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png"
            )
        )

        paths.extend(files)
        labels.extend(
            [class_id] * len(files)
        )

    return paths, np.asarray(
        labels,
        dtype=np.int32
    )


def load_image(path):
    # Sama seperti preprocessing training Model J.
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


# ==========================================
# ROTATION PROCESSING
# ==========================================

def rotate_image(image, angle, method):
    if angle == 0:
        return image.copy()

    # Rotasi kelipatan 90 tetap persis.
    if angle in (90, 180, 270):
        return np.rot90(
            image,
            k=angle // 90
        ).copy()

    height, width = image.shape[:2]

    center = (
        (width - 1) / 2.0,
        (height - 1) / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    border_mode = FILL_METHODS[method]

    if method == "white_fill":
        border_value = (255, 255, 255)
    else:
        border_value = (0, 0, 0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value
    )

    return rotated.astype(np.float32)


# ==========================================
# METRICS
# ==========================================

def get_metrics(actual, probabilities):
    predicted = probabilities.argmax(axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    fresh_total = int(
        np.sum(actual == 0)
    )

    rotten_total = int(
        np.sum(actual == 1)
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "fresh_recall": float(
            matrix[0, 0] / fresh_total
        ),
        "rotten_recall": float(
            matrix[1, 1] / rotten_total
        ),
        "fresh_to_rotten": int(matrix[0, 1]),
        "rotten_to_fresh": int(matrix[1, 0]),
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
# CREATE VISUAL PREVIEW
# ==========================================

def create_preview(images, paths, labels):
    selected = []

    for class_id in (0, 1):
        indexes = np.where(labels == class_id)[0]
        selected.extend(indexes[:2].tolist())

    methods = list(FILL_METHODS.keys())
    angles = [45, 135]

    cell_w = 230
    cell_h = 260

    columns = len(methods)
    rows = len(selected) * len(angles)

    sheet = Image.new(
        "RGB",
        (cell_w * columns, cell_h * rows),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    row_index = 0

    for image_index in selected:
        for angle in angles:
            y = row_index * cell_h

            for col, method in enumerate(methods):
                rotated = rotate_image(
                    images[image_index],
                    angle,
                    method
                )

                picture = Image.fromarray(
                    np.clip(
                        rotated,
                        0,
                        255
                    ).astype(np.uint8)
                )

                x = col * cell_w

                sheet.paste(
                    picture,
                    (x + 3, y + 3)
                )

                draw.text(
                    (x + 5, y + 231),
                    f"{method} | {angle} deg",
                    fill="black"
                )

            draw.text(
                (5, y + 246),
                f"{CLASS_NAMES[labels[image_index]]}: "
                f"{paths[image_index].name[:28]}",
                fill="black"
            )

            row_index += 1

    output = REPORT_DIR / "rotation_preview.jpg"
    sheet.save(output, quality=95)

    print("Preview saved:", output)


# ==========================================
# MAIN
# ==========================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    paths, labels = get_files()

    if len(paths) == 0:
        raise ValueError("Dataset validation kosong.")

    images = np.stack([
        load_image(path)
        for path in paths
    ]).astype(np.float32)

    print("\n=== MODEL J ROTATION DIAGNOSTIC ===")
    print("Images:", len(images))
    print("Classes:", CLASS_NAMES)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    summary_rows = []
    prediction_rows = []
    confusion_data = {}

    for method in FILL_METHODS:
        print(f"\n=== {method.upper()} ===")

        confusion_data[method] = {}

        for angle in ANGLES:
            rotated_images = np.stack([
                rotate_image(image, angle, method)
                for image in images
            ]).astype(np.float32)

            probabilities = model.predict(
                rotated_images,
                batch_size=BATCH_SIZE,
                verbose=0
            )

            predicted = probabilities.argmax(axis=1)

            metrics = get_metrics(
                labels,
                probabilities
            )

            summary_rows.append({
                "method": method,
                "angle": angle,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                }
            })

            confusion_data[method][str(angle)] = (
                metrics["confusion_matrix"]
            )

            for i, path in enumerate(paths):
                prediction_rows.append({
                    "method": method,
                    "angle": angle,
                    "filename": str(
                        path.relative_to(VALID_DIR)
                    ),
                    "actual": CLASS_NAMES[labels[i]],
                    "prediction": CLASS_NAMES[predicted[i]],
                    "prob_fresh": float(
                        probabilities[i, 0]
                    ),
                    "prob_rotten": float(
                        probabilities[i, 1]
                    )
                })

            print(
                f"{angle:3d} deg | "
                f"Accuracy: {metrics['accuracy']:.2%} | "
                f"Rotten recall: "
                f"{metrics['rotten_recall']:.2%} | "
                f"Rotten -> Fresh: "
                f"{metrics['rotten_to_fresh']}"
            )

    save_csv(
        REPORT_DIR / "rotation_diagnostic.csv",
        summary_rows
    )

    save_csv(
        REPORT_DIR / "rotation_predictions.csv",
        prediction_rows
    )

    with open(
        REPORT_DIR / "confusion_matrices.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            confusion_data,
            file,
            indent=2
        )

    create_preview(images, paths, labels)

    # Rata-rata accuracy setiap metode.
    for method in FILL_METHODS:
        results = [
            row["accuracy"]
            for row in summary_rows
            if row["method"] == method
        ]

        print(
            f"\nMean accuracy {method}: "
            f"{np.mean(results):.2%}"
        )

    print("\n=== DIAGNOSTIC FINISHED ===")
    print("Reports saved:", REPORT_DIR)


if __name__ == "__main__":
    main()
