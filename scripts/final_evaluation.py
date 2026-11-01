
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocessing import load_dataset, CLASS_NAMES

MODEL_PATH = Path("models/rotation_focused.keras")
DATASET_DIR = Path("dataset/cleaned/test")
REPORT_DIR = Path("reports/final")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
ANGLES = [0, 45, 90, 135, 180, 270]
BATCH_SIZE = 16

print("=== FINAL EVALUATION ===")

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ======================================
# 1. EVALUASI TEST SET ASLI
# ======================================

test_ds = load_dataset(
    "test",
    shuffle=False
)

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision_macro": precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    ),
    "recall_macro": recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    ),
    "f1_macro": f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )
}

print("\n=== HASIL TEST SET ===")

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

with open(
    REPORT_DIR / "final_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=2)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).T.to_csv(
    REPORT_DIR / "final_classification_report.csv"
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Final Confusion Matrix - Rotation Focused")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "final_confusion_matrix.png",
    dpi=150
)

plt.close()

# ======================================
# 2. PENGUJIAN ROTASI TEST SET
# ======================================

samples = []

for label_id, label in enumerate(CLASS_NAMES):
    folder = DATASET_DIR / label

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in (
            ".jpg", ".jpeg", ".png"
        ):
            samples.append((path, label_id))

rotation_true = np.array([
    label for _, label in samples
])


def prepare_image(path, angle):
    if angle == 0:
        raw = tf.io.read_file(str(path))

        image = tf.io.decode_image(
            raw,
            channels=3,
            expand_animations=False
        )

        return tf.image.resize(
            image,
            IMG_SIZE,
            method="bilinear"
        ).numpy().astype(np.float32)

    with Image.open(path) as img:
        rotated = img.convert("RGB").rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

        array = np.asarray(
            rotated,
            dtype=np.float32
        )

    return tf.image.resize(
        array,
        IMG_SIZE,
        method="bilinear"
    ).numpy().astype(np.float32)


rotation_results = []
rotation_details = []

for angle in ANGLES:
    all_probabilities = []

    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start:start + BATCH_SIZE]

        images = np.stack([
            prepare_image(path, angle)
            for path, _ in batch
        ])

        batch_probabilities = model.predict(
            images,
            verbose=0
        )

        all_probabilities.append(batch_probabilities)

    probs = np.concatenate(
        all_probabilities,
        axis=0
    )

    predictions = np.argmax(probs, axis=1)

    accuracy = accuracy_score(
        rotation_true,
        predictions
    )

    f1 = f1_score(
        rotation_true,
        predictions,
        average="macro",
        zero_division=0
    )

    rotation_results.append({
        "angle": angle,
        "accuracy": accuracy,
        "f1_macro": f1
    })

    for i, (path, label_id) in enumerate(samples):
        rotation_details.append({
            "filename": path.name,
            "angle": angle,
            "actual": CLASS_NAMES[label_id],
            "predicted": CLASS_NAMES[predictions[i]],
            "confidence": float(np.max(probs[i]))
        })

    print(
        f"Rotasi {angle}°: "
        f"Accuracy={accuracy:.4f}, F1={f1:.4f}"
    )

pd.DataFrame(rotation_results).to_csv(
    REPORT_DIR / "final_rotation_results.csv",
    index=False
)

pd.DataFrame(rotation_details).to_csv(
    REPORT_DIR / "final_rotation_predictions.csv",
    index=False
)

print("\nFinal evaluation selesai!")
print(f"Hasil tersimpan di: {REPORT_DIR}")
