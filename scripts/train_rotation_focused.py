
from pathlib import Path
import json

import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import load_dataset

tf.keras.utils.set_random_seed(42)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

BASELINE_PATH = MODEL_DIR / "baseline.keras"
OUTPUT_PATH = MODEL_DIR / "rotation_focused.keras"

EPOCHS = 20

# Rotasi acak dalam rentang -180 hingga +180 derajat
rotation = tf.keras.layers.RandomRotation(
    factor=0.5,
    fill_mode="constant",
    fill_value=0.0
)


def augment_batch(images, labels):
    rotated = rotation(images, training=True)

    # Setiap gambar memiliki peluang 50% untuk dirotasi.
    # Sisanya tetap dalam orientasi asli.
    mask = tf.random.uniform(
        shape=(tf.shape(images)[0], 1, 1, 1)
    ) < 0.5

    output = tf.where(mask, rotated, images)

    return output, labels


def create_training_dataset():
    train_ds = load_dataset(
        "train",
        shuffle=True
    )

    train_ds = train_ds.map(
        augment_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_ds.prefetch(tf.data.AUTOTUNE)


def save_graph(history):
    for metric in ["accuracy", "loss"]:
        plt.figure(figsize=(9, 5))

        plt.plot(
            history.history[metric],
            label=f"Training {metric}"
        )

        plt.plot(
            history.history[f"val_{metric}"],
            label=f"Validation {metric}"
        )

        plt.title(f"Rotation Focused - {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / f"rotation_focused_{metric}.png",
            dpi=150
        )

        plt.close()


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)

    print("=== TRAINING ROTATION FOCUSED ===")

    train_ds = create_training_dataset()
    valid_ds = load_dataset("valid")

    model = tf.keras.models.load_model(
        BASELINE_PATH,
        compile=False
    )

    # Fine-tuning classification head.
    # Backbone MobileNetV2 tetap frozen sesuai baseline.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_PATH),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    save_graph(history)

    with open(
        REPORT_DIR / "rotation_focused_history.json",
        "w"
    ) as f:
        json.dump(history.history, f, indent=2)

    best_model = tf.keras.models.load_model(
        OUTPUT_PATH,
        compile=False
    )

    best_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    loss, accuracy = best_model.evaluate(valid_ds)

    print("\n=== HASIL MODEL D ===")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Model: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
