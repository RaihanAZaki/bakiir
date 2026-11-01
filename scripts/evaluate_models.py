
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocessing import load_dataset, CLASS_NAMES

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    "baseline",
    "augmentation",
    "augmentation_light",
]

# Gunakan validation set untuk perbandingan
valid_ds = load_dataset("valid", shuffle=False)

y_true = np.concatenate([
    labels.numpy()
    for _, labels in valid_ds
])

results = []

for name in MODEL_NAMES:
    model_path = MODEL_DIR / f"{name}.keras"

    if not model_path.exists():
        print(f"Model tidak ditemukan: {model_path}")
        continue

    print(f"\n=== EVALUASI {name.upper()} ===")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    probabilities = model.predict(
        valid_ds,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{name}_confusion_matrix.png",
        dpi=150
    )

    plt.close()

    pd.DataFrame(report).T.to_csv(
        REPORT_DIR / f"{name}_classification_report.csv"
    )

    predictions = pd.DataFrame({
        "actual": [CLASS_NAMES[i] for i in y_true],
        "predicted": [CLASS_NAMES[i] for i in y_pred],
        "confidence": np.max(probabilities, axis=1),
    })

    predictions.to_csv(
        REPORT_DIR / f"{name}_predictions.csv",
        index=False
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
    })

    print(f"Accuracy: {accuracy:.4f}")

comparison = pd.DataFrame(results)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n=== PERBANDINGAN MODEL ===")
print(comparison.to_string(index=False))

print("\nEvaluasi selesai.")
