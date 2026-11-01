
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

DATASET_DIR = Path("dataset/cleaned/valid")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

CLASSES = ["Fresh", "Half-Fresh", "Rotten"]
ANGLES = [0, 45, 90, 135, 180, 270]

MODELS = [
    "finetuning",
    "finetuning_advanced",
    "rotation_robust"
]


IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Simpan semua gambar validasi
samples = []

for label_id, label in enumerate(CLASSES):
    folder = DATASET_DIR / label

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in [
            ".jpg", ".jpeg", ".png"
        ]:
            samples.append((path, label_id))

print(f"Total gambar validasi: {len(samples)}")



def prepare_image(path, angle):
    raw = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False
    )

    # Untuk sudut 0, gunakan preprocessing yang sama
    # dengan image_dataset_from_directory
    if angle == 0:
        image = tf.image.resize(
            image,
            IMG_SIZE,
            method="bilinear"
        )

        return image.numpy().astype(np.float32)

    # Rotasi gambar menggunakan resolusi asli
    pil_image = Image.open(path).convert("RGB")

    try:
        rotated_image = pil_image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

        image_array = np.asarray(
            rotated_image,
            dtype=np.float32
        )
    finally:
        pil_image.close()

    # Resize setelah rotasi
    image = tf.image.resize(
        image_array,
        IMG_SIZE,
        method="bilinear"
    )

    return image.numpy().astype(np.float32)



results = []
details = []

y_true = np.array([
    label for _, label in samples
])

for model_name in MODELS:
    print(f"\n=== MODEL {model_name.upper()} ===")

    model = tf.keras.models.load_model(
        MODEL_DIR / f"{model_name}.keras",
        compile=False
    )

    for angle in ANGLES:
        predictions = []
        confidences = []

        for start in range(
            0, len(samples), BATCH_SIZE
        ):
            batch_samples = samples[
                start:start + BATCH_SIZE
            ]

            images = np.stack([
                prepare_image(path, angle)
                for path, _ in batch_samples
            ])

            probabilities = model.predict(
                images,
                verbose=0
            )

            predictions.extend(
                np.argmax(
                    probabilities, axis=1
                ).tolist()
            )

            confidences.extend(
                np.max(
                    probabilities, axis=1
                ).tolist()
            )

        accuracy = accuracy_score(
            y_true, predictions
        )

        f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "model": model_name,
            "angle": angle,
            "accuracy": accuracy,
            "f1_macro": f1
        })

        for i, (path, label_id) in enumerate(samples):
            details.append({
                "model": model_name,
                "angle": angle,
                "filename": path.name,
                "actual": CLASSES[label_id],
                "predicted": CLASSES[predictions[i]],
                "confidence": confidences[i]
            })

        print(
            f"Rotasi {angle:3d}° | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f}"
        )

pd.DataFrame(results).to_csv(
    REPORT_DIR / "rotation_results.csv",
    index=False
)

pd.DataFrame(details).to_csv(
    REPORT_DIR / "rotation_predictions.csv",
    index=False
)

print("\n=== PERBANDINGAN ACCURACY ===")

comparison = pd.DataFrame(results).pivot(
    index="angle",
    columns="model",
    values="accuracy"
)

print(comparison.to_string())
print("\nPengujian rotasi selesai!")
